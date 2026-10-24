"""Chạy bằng interpreter AV-HuBERT riêng; source chính thức + checkpoint Base hoặc Large.

Không nội suy landmark qua khoảng mất mặt. Mỗi valid run/chunk được encoder
xử lý độc lập; feature ở frame thiếu giữ zero và mask=False.

Hai giai đoạn (tách 24/10: dlib dò mặt chiếm ~86 % thời gian, chỉ cần CPU):
- stage "landmarks": giải mã → dlib 68 điểm → hộp miệng / mặt nạ hợp lệ → `landmarks/<id>.npz`.
  Chạy được trên máy không GPU, không nạp torch / fairseq.
- stage "full": đọc landmark có sẵn (request["landmarks"]) hoặc tự dò bằng dlib (demo) → ROI môi →
  filterbank → hai lượt encoder → `avhubert/<id>.npz`. Hai cách cho cùng một kết quả: cùng hàm
  `detect` / `geometry`.

Một lần: `python avhubert_worker.py request.json`. Chạy liên tục: `--serve <config JSON>`,
nạp một lần rồi nhận từng yêu cầu (video, output, signature, source_fingerprint) qua stdin.
"""

import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from vn_av_df.common.runtime import read_json, save_npz, sha, write_json

MOUTH_BOX_SCALE = 1.6  # Hộp vuông quanh landmark 48-67, nới để gồm môi/răng/viền ghép.
CROP = 88  # ROI môi của AV-HuBERT (image_crop_size); stage full kiểm lại theo task.


def runs(mask):
    """Khoảng quan sát liên tục theo frame 25 Hz."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


# Landmark theo nội dung frame (hash điểm ảnh): real, sham và phần ngoài đoạn partial của cùng
# parent có frame giống hệt nhau nên dlib (phần chậm nhất, chạy CPU) không phải dò lại.
LANDMARK_CACHE_FRAMES = 20000


def load_landmarks(cfg):
    """Chỉ dlib + mean face (stage landmarks: không torch / fairseq / GPU)."""
    import dlib

    return dict(
        cfg=cfg,
        detector=dlib.get_frontal_face_detector(),
        predictor=dlib.shape_predictor(cfg["landmarks"]),
        mean_face=np.load(cfg["mean_face"], allow_pickle=False),
        landmarks=collections.OrderedDict(),
    )


def load(cfg):
    """Nạp fairseq, checkpoint và dlib một lần."""
    import torch  # noqa: F401 -- nạp trước fairseq

    state = load_landmarks(cfg)
    upstream = Path(cfg["repo"]).resolve()
    sys.path[:0] = [str(upstream / "avhubert"), str(upstream / "fairseq")]
    # Module AV-HuBERT chọn kiểu import theo len(sys.argv): == 1 → import tuyệt đối (chạy rời);
    # khác → import tương đối (như package), lỗi khi nạp từ thư mục avhubert/. Worker được gọi
    # kèm request.json nên tạm để argv một phần tử trong lúc import.
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        import fairseq
        import hubert  # noqa: F401 -- đăng ký model vào fairseq
        import hubert_asr  # noqa: F401 -- đăng ký av_hubert_seq2seq (checkpoint fine-tune VSR)
        import hubert_pretraining  # noqa: F401 -- đăng ký task
    finally:
        sys.argv = argv
    models, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cfg["checkpoint"]])
    model = models[0]
    # Checkpoint fine-tune (VSR/AVSR, vd. self_large_vox_433h của AVH-Align) bọc AVHubertModel trong
    # encoder.w2v_model; chỉ dùng phần encoder đó. Kích thước 768 (Base) hoặc 1024 (Large).
    if hasattr(model, "decoder"):
        if not cfg.get("allow_finetuned", False):
            raise ValueError("Fine-tuned AV-HuBERT checkpoint: set allow_finetuned in the encoder")
        model = model.encoder.w2v_model
    if model.encoder_embed_dim not in (768, 1024):
        raise ValueError("Unexpected AV-HuBERT width")
    # Tiếng: logfbank 26 chiều xếp 4 khung (104) như lúc pretrain. Checkpoint VSR fine-tune chỉ trên
    # hình nên task ghi stack_order_audio = 1; kiểm kích thước đầu vào thật của encoder tiếng
    # (AVH-Align cũng cố định stack 4, deepfake_feature_extraction.py ở commit 55401de).
    proj = getattr(getattr(model, "feature_extractor_audio", None), "proj", None)
    audio_dim = int(proj.in_features) if proj is not None else 104
    crop = int(task.cfg.image_crop_size)
    if crop != CROP or audio_dim != 104:
        raise ValueError(
            f"Unsupported AV-HuBERT preprocessing: crop {crop}, audio input {audio_dim} "
            "(expected crop 88, audio 26 x 4 = 104)"
        )
    device = cfg.get("device", "cpu")
    model.to(device).eval().requires_grad_(False)
    return dict(state, model=model, task=task, device=device)


def face_landmarks(state, frame):
    """68 landmark của đúng một mặt, hoặc None; frame giống hệt đã gặp thì dùng lại kết quả."""
    import cv2

    key = hashlib.blake2b(frame.tobytes(), digest_size=16).digest() + str(frame.shape).encode()
    cache = state["landmarks"]
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = state["detector"](gray, 1)
    points = None
    if len(faces) == 1:
        shape = state["predictor"](gray, faces[0])
        points = np.array([(shape.part(j).x, shape.part(j).y) for j in range(68)], np.float32)
    cache[key] = points
    if len(cache) > LANDMARK_CACHE_FRAMES:
        cache.popitem(last=False)
    return points


def decode_clip(cfg, video):
    from vn_av_df.data.media import decode

    return decode(video, cfg.get("max_duration", 60), cfg.get("max_side", 640), sample_rate=16000)


def detect(state, decoded):
    """(landmarks float32 [T, 68, 2], hợp lệ trước khi cắt ROI) — phần chậm, chỉ CPU."""
    frames = decoded["frames"]
    landmarks = np.zeros((len(frames), 68, 2), np.float32)
    valid = decoded["audio_valid"] & decoded["visual_valid"]
    for i, frame in enumerate(frames):
        points = face_landmarks(state, frame)
        if points is None:
            valid[i] = False
            continue
        landmarks[i] = points
    return landmarks, valid


def audio_frames(pcm):
    """Logfbank 26 chiều xếp 4 khung (104) theo lưới 40 ms của video."""
    from python_speech_features import logfbank

    # 25 ms filterbank, hop10 ms; stack4 khớp grid 40 ms của video.
    # Upstream đọc PCM int16; giữ cùng thang biên độ trước log-filterbank.
    audio = logfbank(np.clip(pcm * 32768, -32768, 32767), samplerate=16000).astype(np.float32)
    return np.pad(audio, ((0, (-len(audio)) % 4), (0, 0))).reshape(-1, 104)


def geometry(frames, landmarks, valid, mean_face, n_audio):
    """Làm mượt landmark trong từng đoạn hợp lệ, căn theo mean face upstream → ROI môi uint8,
    hộp miệng [cx, cy, side] trên frame đã decode, mặt nạ hợp lệ cuối (thêm điều kiện căn / cắt
    và độ dài tiếng). Dùng chung cho hai stage nên kết quả giống hệt."""
    import cv2
    from skimage import transform as tf

    valid = valid.copy()
    patches = np.zeros((len(frames), CROP, CROP), np.uint8)
    mouth_boxes = np.zeros((len(frames), 3), np.float32)
    stable = [33, 36, 39, 42, 45]
    for left, right in list(runs(valid)):
        for i in range(left, right):
            smooth = landmarks[max(left, i - 6) : min(right, i + 7)].mean(0)
            low, high = smooth[48:68].min(0), smooth[48:68].max(0)
            mouth_boxes[i] = [*((low + high) / 2), (high - low).max() * MOUTH_BOX_SCALE]
            transform = tf.SimilarityTransform()
            if not transform.estimate(smooth[stable], mean_face[stable]):
                valid[i] = False
                continue
            aligned = cv2.warpAffine(frames[i], transform.params[:2], (256, 256))
            center = transform(landmarks[i])[48:68].mean(0)
            x, y = np.rint(center - 48).astype(int)
            if x < 0 or y < 0 or x + 96 > 256 or y + 96 > 256:
                valid[i] = False
                continue
            patches[i] = cv2.cvtColor(aligned[y : y + 96, x : x + 96], cv2.COLOR_BGR2GRAY)[
                4:92, 4:92
            ]
    # Như bản một-stage cũ: cắt theo độ dài tiếng sau khi làm mượt / cắt ROI.
    if abs(n_audio - len(frames)) > 1:
        raise ValueError("Audio/video feature timelines differ by more than one frame")
    valid[n_audio:] = False
    return patches, mouth_boxes, valid


def extract_landmarks(state, request):
    """Stage landmarks (CPU): file đủ cho mouth / lipinc (hộp miệng, landmark, mặt nạ) và cho
    stage full (landmark float32 + mặt nạ trước khi cắt ROI)."""
    cfg = state["cfg"]
    decoded = decode_clip(cfg, request["video"])
    frames, pcm = decoded["frames"], decoded["pcm"]
    landmarks, detected = detect(state, decoded)
    _, mouth_boxes, valid = geometry(
        frames, landmarks, detected, state["mean_face"], len(audio_frames(pcm))
    )
    output = Path(request["output"])
    save_npz(
        output,
        points=landmarks,
        detected_valid=detected,
        times_s=np.arange(len(frames)) / 25,
        audio_valid=valid,
        visual_valid=valid,
        mouth_boxes=mouth_boxes,
        landmarks=landmarks.astype(np.float16),
    )
    write_json(
        output.with_suffix(".json"),
        dict(
            format="avhubert-landmarks-v1",
            step_s=0.04,
            frames=len(frames),
            duration_s=len(frames) / 25,
            origin_s=decoded["origin_s"],
            valid_ratio=float(valid.mean()) if len(valid) else 0.0,
            feature_signature=request["signature"],
            feature_sha256=sha(output),
            source_fingerprint=request["source_fingerprint"],
            preprocessing="dlib frontal (upsample 1) + 68-point predictor; smoothing13; mouth-box1.6",
        ),
    )


def extract(state, request):
    """Stage full: (landmark có sẵn hoặc dlib) → ROI môi → filterbank → hai lượt encoder → cache."""
    import torch
    from torch.nn import functional as F

    cfg, model, task, device = state["cfg"], state["model"], state["task"], state["device"]
    decoded = decode_clip(cfg, request["video"])
    frames, pcm = decoded["frames"], decoded["pcm"]
    if request.get("landmarks"):
        with np.load(request["landmarks"], allow_pickle=False) as z:
            landmarks, detected = z["points"].astype(np.float32), z["detected_valid"].copy()
        if len(landmarks) != len(frames):
            raise ValueError("Landmark cache has a different frame count; re-run stage landmarks")
        detected &= decoded["audio_valid"] & decoded["visual_valid"]
    else:
        landmarks, detected = detect(state, decoded)
    audio = audio_frames(pcm)
    patches, mouth_boxes, valid = geometry(
        frames, landmarks, detected, state["mean_face"], len(audio)
    )
    rois = np.where(
        valid[:, None, None],
        (patches / 255.0 - task.cfg.image_mean) / task.cfg.image_std,
        0.0,
    ).astype(np.float32)
    if len(audio) < len(frames):
        audio = np.pad(audio, ((0, len(frames) - len(audio)), (0, 0)))
    audio = torch.from_numpy(audio[: len(frames)])
    if task.cfg.normalize:
        audio = F.layer_norm(audio, audio.shape[1:])
    dim = model.encoder_embed_dim
    af, vf = np.zeros((len(frames), dim), np.float32), np.zeros((len(frames), dim), np.float32)
    weights = np.zeros(len(frames), np.float32)
    support = np.zeros((len(frames), 2), np.float64)
    chunk, overlap = cfg.get("chunk_frames", 200), cfg.get("overlap_frames", 50)
    if not 0 <= overlap < chunk:
        raise ValueError("Need 0 <= overlap < chunk_frames")
    with torch.no_grad():
        for left, right in runs(valid):
            for start in range(left, right, chunk - overlap):
                end = min(right, start + chunk)
                a = audio[start:end].T[None].to(device)
                v = torch.from_numpy(rois[start:end])[None, None].to(device)
                ax, _ = model.extract_finetune(
                    source={"audio": a, "video": None}, padding_mask=None, output_layer=None
                )
                vx, _ = model.extract_finetune(
                    source={"audio": None, "video": v}, padding_mask=None, output_layer=None
                )
                if ax.shape != (1, end - start, dim) or vx.shape != ax.shape:
                    raise ValueError("Unexpected native feature shape")
                af[start:end] += ax[0].float().cpu().numpy()
                vf[start:end] += vx[0].float().cpu().numpy()
                # Support union của mọi chunk góp vào một native frame.
                first = weights[start:end] == 0
                support[start:end, 0] = np.where(
                    first, start / 25, np.minimum(support[start:end, 0], start / 25)
                )
                support[start:end, 1] = np.maximum(support[start:end, 1], end / 25)
                weights[start:end] += 1
                if end == right:
                    break
    af /= np.maximum(weights[:, None], 1)
    vf /= np.maximum(weights[:, None], 1)
    valid &= weights > 0
    if not np.isfinite(af).all() or not np.isfinite(vf).all():
        raise ValueError("Nonfinite AV-HuBERT output")
    # Đường bao tiếng (RMS mỗi frame 40 ms) và landmark thô (môi cho BioLip / MTN, hộp miệng).
    envelope = np.sqrt(np.square(pcm[: len(frames) * 640].reshape(len(frames), 640)).mean(1))
    output = Path(request["output"])
    save_npz(
        output,
        audio=af,
        visual=vf,
        times_s=np.arange(len(frames)) / 25,
        audio_valid=valid,
        visual_valid=valid,
        support_s=support,
        mouth_boxes=mouth_boxes,
        landmarks=landmarks.astype(np.float16),
        envelope=envelope.astype(np.float32),
    )
    write_json(
        output.with_suffix(".json"),
        dict(
            format="avhubert-native-v2",
            step_s=0.04,
            output_stride=cfg.get("output_stride", 5),
            window_s=chunk / 25,
            duration_s=len(frames) / 25,
            origin_s=decoded["origin_s"],
            feature_signature=request["signature"],
            feature_sha256=sha(output),
            source_fingerprint=request["source_fingerprint"],
            landmarks="cache" if request.get("landmarks") else "dlib",
            preprocessing="68-point stable similarity; per-valid-run smoothing13; ROI96-center88; logfbank-stack4; mouth-box1.6",
        ),
    )


def main(request):
    cfg = request["config"]
    if cfg.get("stage") == "landmarks":
        extract_landmarks(load_landmarks(cfg), request)
    else:
        extract(load(cfg), request)


if __name__ == "__main__":
    if sys.argv[1] == "--serve":
        from vn_av_df.worker import serve

        cfg = json.loads(sys.argv[2])
        if cfg.get("stage") == "landmarks":
            loaded = load_landmarks(cfg)
            serve(lambda request: extract_landmarks(loaded, request))
        else:
            loaded = load(cfg)
            serve(lambda request: extract(loaded, request))
    else:
        main(read_json(sys.argv[1]))
