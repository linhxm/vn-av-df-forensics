import os
import tempfile
from pathlib import Path

import numpy as np

from vn_av_df.common.runtime import run
from vn_av_df.data.media import ffmpeg


def encode(frames, pcm, output, crf):
    import cv2
    from scipy.io import wavfile

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        tmp = Path(tmp)
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(tmp / "video.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 25, (w, h)
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot create lossless intermediate")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        wavfile.write(
            tmp / "audio.wav", 48000, np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
        )
        run(
            [
                ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-i",
                tmp / "video.avi",
                "-i",
                tmp / "audio.wav",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "1",
                "-shortest",
                tmp / "result.mp4",
            ]
        )
        os.replace(tmp / "result.mp4", output)
