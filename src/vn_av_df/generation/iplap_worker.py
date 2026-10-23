"""Chạy inference_single.py của IP-LAP (cwd = repo IP-LAP, python của env gen_iplap).

Upstream giữ nguyên thuật toán; chỉ đổi I/O: upstream ghi sketch|kết quả cạnh nhau rồi mux ffmpeg,
ở đây chỉ giữ nửa kết quả (FFV1, đúng kích thước gốc) và bỏ bước mux. Upstream ghi kết quả đầu
3 lần (frame 0, 1 = frame 2), số frame vẫn khớp. Upstream tách đường dẫn theo '/' và '.', nên
input được chép vào thư mục tương đối không có dấu chấm. Phải chạy được trên Python 3.7.

Upstream chạy FaceMesh trên cả khung và bỏ clip nếu một frame không thấy mặt. Ở part 1, 43/400
job IP-LAP bị bỏ vì vậy, dồn vào mặt nhỏ / nghiêng / có người thứ hai: mẫu IP-LAP lệch về mặt
dễ. Khi đó chạy lại trên một khung cố định quanh mặt cả clip (bộ dò mặt tầm xa của MediaPipe,
khung vuông bao mọi hộp mặt + 1,2 cạnh mặt) rồi dán kết quả về khung gốc: IP-LAP chỉ vẽ lại
vùng mặt, ngoài mặt giữ nguyên điểm ảnh, không có đường nối. Thử trên part 1: 43 clip đó còn 9 %
clip trượt (khung gốc 88 %), 30 clip đối chứng vẫn 0 %. Clip chạy được trên khung gốc giữ y
như trước.

    python iplap_worker.py --video face.mp4 --audio audio.wav --output result.avi
"""

import argparse
import os
import random
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

CROP_SCALE = 2.2


def read_frames(path):
    import cv2

    stream, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, frame = stream.read()
        if not ok:
            stream.release()
            return frames
        frames.append(frame)


def face_window(frames):
    """(trái, trên, cạnh) khung vuông cố định quanh mặt cả clip, hoặc None nếu không thấy mặt."""
    import cv2
    import mediapipe as mp
    import numpy as np

    h, w = frames[0].shape[:2]
    boxes = []
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as detector:
        for frame in frames[::2]:
            found = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).detections
            if found:
                b = max(found, key=lambda d: d.location_data.relative_bounding_box.width)
                b = b.location_data.relative_bounding_box
                boxes.append(
                    [b.xmin * w, b.ymin * h, (b.xmin + b.width) * w, (b.ymin + b.height) * h]
                )
    if not boxes:
        return None
    b = np.array(boxes)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    face = max((b[:, 2] - b[:, 0]).max(), (b[:, 3] - b[:, 1]).max())
    side = int(min(max(x1 - x0, y1 - y0) + (CROP_SCALE - 1) * face, h, w)) // 2 * 2  # yuv420: chẵn
    left = int(np.clip((x0 + x1) / 2 - side / 2, 0, w - side))
    top = int(np.clip((y0 + y1) / 2 - side / 2, 0, h - side))
    return left, top, side


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # face_alignment mới torch.compile FAN; không có triton (Windows) thì lỗi khi chạy.
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import cv2
    import numpy as np
    import torch
    import torchvision

    work, temp = Path(f"iplap{os.getpid()}"), Path(f"tempfile_of_iplap{os.getpid()}")
    work.mkdir()
    shutil.copyfile(args.audio, work / "audio.wav")
    writer = cv2.VideoWriter

    class ResultHalf:
        """Upstream ghi [sketch | kết quả]; chỉ giữ nửa phải, lossless."""

        def __init__(self, path, fourcc, fps, size):
            self.width = size[0] // 2
            self.stream = writer(path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (self.width, size[1]))

        def write(self, frame):
            self.stream.write(np.ascontiguousarray(frame[:, self.width :]))

        def release(self):
            self.stream.release()

    cv2.VideoWriter = ResultHalf
    call = subprocess.call
    # Bỏ lệnh mux cuối (đọc result.avi); các lệnh ffmpeg chuẩn hoá đầu vào vẫn chạy.
    subprocess.call = lambda command, **kw: 0 if "result.avi" in command else call(command, **kw)
    # VGG19 chỉ dùng cho loss khi train và được nạp từ checkpoint renderer: khỏi tải ImageNet;
    # torchvision mới đã bỏ tham số pretrained.
    vgg19 = torchvision.models.vgg19
    torchvision.models.vgg19 = lambda pretrained=False, **kw: vgg19(**kw)
    sys.path.insert(0, os.getcwd())

    def upstream(face):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        shutil.rmtree(temp, ignore_errors=True)
        sys.argv = [
            "inference_single.py",
            *("--input", f"{work.name}/{face}"),
            *("--audio", f"{work.name}/audio.wav", "--output_dir", work.name),
        ]
        runpy.run_path("inference_single.py", run_name="__main__")

    try:
        face = f"face{Path(args.video).suffix}"
        shutil.copyfile(args.video, work / face)
        try:
            upstream(face)
            shutil.copyfile(temp / "result.avi", args.output)
            return
        except NotImplementedError:  # upstream: "raise NotImplementedError  # not detect face"
            frames = read_frames(args.video)
            window = face_window(frames)
            if window is None:
                raise
        left, top, side = window
        print(f"IP-LAP: FaceMesh trượt trên khung gốc, chạy lại trên khung {side}px quanh mặt")
        crop = writer(str(work / "face.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 25, (side, side))
        for frame in frames:
            crop.write(np.ascontiguousarray(frame[top : top + side, left : left + side]))
        crop.release()
        try:
            upstream("face.avi")
        except NotImplementedError:
            print("Face not detected (IP-LAP FaceMesh, cả khi crop quanh mặt)")
            raise
        result = read_frames(temp / "result.avi")
        h, w = frames[0].shape[:2]
        out = writer(args.output, cv2.VideoWriter_fourcc(*"FFV1"), 25, (w, h))
        for i, frame in enumerate(frames):
            if i < len(result):
                frame = frame.copy()
                frame[top : top + side, left : left + side] = result[i]
            out.write(frame)
        out.release()
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()
