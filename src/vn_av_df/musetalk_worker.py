"""Bridge lossless dùng API inference chính thức MuseTalk 1.5, không gọi shell upstream."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vn_av_df.common.runtime import read_json, run


def main(request):
    """Sinh đúng số frame nguồn; fail khi thiếu face/audio, không loop nguồn."""
    import cv2
    import numpy as np
    import torch
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from transformers import WhisperModel

    cfg = request["config"]
    sys.path.insert(0, str(Path(cfg["repo"]).resolve()))
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.blending import get_image
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox
    from musetalk.utils.utils import datagen, load_all_model

    from vn_av_df.data.media import decode, ffmpeg

    device = torch.device(cfg.get("device", "cuda"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    n, w, h = request["frames"], request["width"], request["height"]
    source = decode(request["video"], max_side=max(w, h), sample_rate=48000)
    donor = decode(request["audio_video"], max_side=max(w, h), sample_rate=48000)
    if len(source["frames"]) != n or len(donor["pcm"]) < n * 1920:
        raise ValueError("MuseTalk source/donor timeline differs from plan")
    vae, unet, pe = load_all_model(
        unet_model_path="models/musetalkV15/unet.pth",
        unet_config="models/musetalkV15/musetalk.json",
        device=device,
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae.vae.to(device=device, dtype=dtype).eval()
    unet.model.to(device=device, dtype=dtype).eval()
    pe.to(device=device, dtype=dtype).eval()
    whisper = (
        WhisperModel.from_pretrained("models/whisper", local_files_only=True)
        .to(device=device, dtype=dtype)
        .eval()
    )
    processor, parser = (
        AudioProcessor(feature_extractor_path="models/whisper"),
        FaceParsing(left_cheek_width=90, right_cheek_width=90),
    )
    with tempfile.TemporaryDirectory(dir=Path(request["output"]).parent) as folder:
        folder = Path(folder)
        images = []
        for i, frame in enumerate(source["frames"]):
            path = folder / f"{i:08d}.png"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError("Failed to write input frame")
            images.append(str(path))
        audio = donor["pcm"][: n * 1920]
        wavfile.write(folder / "audio16.wav", 16000, resample_poly(audio, 1, 3).astype(np.float32))
        wavfile.write(
            folder / "audio48.wav", 48000, (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        )
        coords, frames = get_landmark_and_bbox(images, 0)
        if len(coords) != n or any(tuple(b) == tuple(coord_placeholder) for b in coords):
            raise ValueError("Missing MuseTalk face; cannot label full clip as manipulated")
        latents, boxes = [], []
        with torch.no_grad():
            for frame, box in zip(frames, coords):
                x1, y1, x2, y2 = map(int, box)
                y2 = min(y2 + 10, h)
                if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
                    raise ValueError("Invalid face bbox")
                boxes.append((x1, y1, x2, y2))
                patch = cv2.resize(
                    frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4
                )
                latents.append(vae.get_latents_for_unet(patch))
            inputs, length = processor.get_audio_feature(str(folder / "audio16.wav"))
            chunks = processor.get_whisper_chunk(
                inputs,
                device,
                dtype,
                whisper,
                length,
                fps=25,
                audio_padding_length_left=2,
                audio_padding_length_right=2,
            )
            if len(chunks) < n:
                raise ValueError("Whisper produced fewer frames than source")
            writer = cv2.VideoWriter(
                str(folder / "video.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 25, (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError("Cannot write FFV1")
            index = 0
            try:
                for audio_batch, latent_batch in datagen(
                    chunks[:n], latents, cfg.get("batch_size", 4), device=device
                ):
                    features = pe(audio_batch.to(device=device, dtype=dtype))
                    predicted = unet.model(
                        latent_batch.to(device=device, dtype=dtype),
                        torch.tensor([0], device=device),
                        encoder_hidden_states=features,
                    ).sample
                    for face in vae.decode_latents(predicted):
                        x1, y1, x2, y2 = boxes[index]
                        face = cv2.resize(face.astype(np.uint8), (x2 - x1, y2 - y1))
                        composite = get_image(
                            frames[index].copy(), face, [x1, y1, x2, y2], mode="jaw", fp=parser
                        )
                        writer.write(composite)
                        index += 1
            finally:
                writer.release()
            if index != n:
                raise ValueError("Generated frame count differs")
        run(
            [
                ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-i",
                folder / "video.avi",
                "-i",
                folder / "audio48.wav",
                "-c:v",
                "ffv1",
                "-c:a",
                "pcm_s16le",
                "-t",
                n / 25,
                request["output"],
            ]
        )


if __name__ == "__main__":
    main(read_json(sys.argv[1]))
