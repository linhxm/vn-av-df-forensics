"""Chạy inference.py của VideoRetalking (cwd = repo, python của env gen_vrt), cùng tham số.

Upstream giữ nguyên; chỉ thêm đường dự phòng cho bước Croper: dlib HOG tìm mặt ở nửa đầu clip để
căn khung kiểu FFHQ, không frame nào thấy thì crop() trả None và upstream dừng bằng
"TypeError: cannot unpack non-iterable NoneType object". Ở part 1 cả 4 lỗi này rơi vào một người
(mặt nhìn thẳng, ~240 px, dlib vẫn trượt 100 % frame, crop quanh mặt cũng không giúp). Khi dlib
không thấy mặt, lấy hộp mặt từ S3FD (face_alignment, chính bộ dò upstream dùng ở các bước sau)
rồi vẫn đặt landmark bằng shape predictor 68 điểm của dlib như upstream. Frame dlib thấy mặt
giữ y như cũ. S3FD cũng trượt thì báo "Face not detected" để pipeline bỏ job vì lý do mặt.

    python vrt_worker.py --face face.mp4 --audio audio.wav --outfile result.mp4 ...
"""

import os
import runpy
import sys


def main():
    sys.path.insert(0, os.getcwd())
    import dlib
    import numpy as np
    from utils import ffhq_preprocess

    original = ffhq_preprocess.Croper.get_landmark
    detector = []

    def get_landmark(self, img_np):
        lm = original(self, img_np)
        if lm is not None:
            return lm
        if not detector:
            import torch
            from face_alignment.detection.sfd import FaceDetector

            detector.append(FaceDetector("cuda" if torch.cuda.is_available() else "cpu"))
        # img_np là RGB (Croper nhận frame RGB); S3FD của face_alignment cũng nhận RGB.
        boxes = [b for b in detector[0].detect_from_image(img_np.copy()) if b[4] > 0.5]
        if not boxes:
            return None
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[:4]
        shape = self.predictor(img_np, dlib.rectangle(int(x1), int(y1), int(x2), int(y2)))
        return np.array([[p.x, p.y] for p in shape.parts()])

    ffhq_preprocess.Croper.get_landmark = get_landmark
    crop = ffhq_preprocess.Croper.crop

    def checked_crop(self, *args, **kwargs):
        result = crop(self, *args, **kwargs)
        if result is None:
            raise ValueError("Face not detected (VideoRetalking Croper: dlib và S3FD)")
        return result

    ffhq_preprocess.Croper.crop = checked_crop
    sys.argv = ["inference.py", *sys.argv[1:]]
    runpy.run_path("inference.py", run_name="__main__")


if __name__ == "__main__":
    main()
