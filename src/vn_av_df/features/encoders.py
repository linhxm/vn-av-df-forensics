"""Encoder của v003 (ngoài AV-HuBERT đã có): XLS-R, crop miệng, khung LIPINC-V2, waveform.

Mọi cache là npz + json cùng tên sample, lưới 25 Hz (`times_s`), kèm `feature_signature`
(asset + code + tuỳ chọn) và `source_fingerprint` (hash video) để không dùng nhầm cache cũ.

- xlsr: facebook/wav2vec2-xls-r-300m (128 ngôn ngữ, có tiếng Việt), trung bình hidden state theo
  từng nhóm lớp `layer_groups` (mặc định 1-4, 5-8, 9-12; Lee et al. 2022: lớp 5 tốt nhất cho
  phát hiện giả), 50 Hz → 25 Hz (trung bình từng cặp) → `ssl` [T, nhóm, 1024] float16. M1 học
  trọng số softmax trên các nhóm.
- mouth: hộp miệng + landmark dlib từ cache AV-HuBERT (khung ≤ 640 px) quy về khung gốc 720p,
  crop xám `crop` × `crop` (mặc định 96, không phóng lên 224) → `mouth` [T, crop, crop] uint8;
  `lips` [T, 20, 2] landmark môi (48-67) chuẩn hoá theo tâm miệng và bề rộng hai mép;
  `pose` [T, 3] yaw / pitch / roll (độ / 90) từ solvePnP trên 6 landmark (lệch môi–đầu, LipDA).
- lipinc: 8 crop miệng màu 64×144 theo cách chọn của LIPINC-V2 (5 frame liên tiếp miệng mở +
  3 frame cùng tư thế miệng), xem `LipincEncoder`.
- wave: tiếng 16 kHz mono xếp theo frame [T, 640] float16 cho XLS-R + AASIST fine-tune.
"""

from pathlib import Path

import numpy as np

from vn_av_df.common.runtime import fingerprint, read_json, require_file, save_npz, sha, write_json


def source_fingerprint(video):
    return fingerprint({"assets": {"video": sha(video)}, "variant": {"kind": "clean"}})


def fresh(output, signature, source):
    meta_path = Path(output).with_suffix(".json")
    if Path(output).is_file() and meta_path.is_file():
        meta = read_json(meta_path)
        return (
            meta.get("feature_signature") == signature
            and meta.get("source_fingerprint") == source
            and meta.get("feature_sha256") == sha(output)
        )
    return False


def audio_16k(video):
    import av

    chunks = []
    with av.open(str(video)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


class XLSREncoder:
    kind = "xlsr"

    def __init__(self, cfg, model=None):
        self.cfg = dict(cfg)
        self.groups = [tuple(g) for g in cfg.get("layer_groups", [(1, 4), (5, 8), (9, 12)])]
        self.device = cfg.get("device", "cpu")
        if model is None:
            import torch
            from transformers import Wav2Vec2Model

            folder = Path(cfg["model"])
            weights = folder / "model.safetensors"
            if not weights.is_file():
                weights = folder / "pytorch_model.bin"
            assets = {
                "config": sha(require_file(folder / "config.json", "XLS-R config")),
                "weights": sha(require_file(weights, "XLS-R weights")),
            }
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            print(f"XLS-R: nạp {folder.name} lên {self.device} ({dtype})…", flush=True)
            model = Wav2Vec2Model.from_pretrained(str(folder), torch_dtype=dtype)
        else:
            assets = {"model": "test-fixture"}
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.signature = fingerprint(
            {
                "assets": assets,
                "code": sha(__file__),
                "layer_groups": [list(g) for g in self.groups],
                "format": "xlsr-25hz-v2",
            }
        )

    def extract(self, request, output):
        import torch

        video = Path(request["video"])
        source = source_fingerprint(video)
        if fresh(output, self.signature, source):
            return read_json(Path(output).with_suffix(".json"))
        pcm = audio_16k(video)
        frames = int(round(len(pcm) / 640))
        x = torch.from_numpy((pcm - pcm.mean()) / (pcm.std() + 1e-7))[None]
        x = x.to(self.device, dtype=next(self.model.parameters()).dtype)
        with torch.no_grad():
            hidden = self.model(x, output_hidden_states=True).hidden_states
        # hidden_states[0] là đầu ra lớp chiếu đặc trưng; hidden_states[i] là lớp transformer i.
        groups = torch.stack(
            [torch.stack(hidden[low : high + 1]).mean(0)[0] for low, high in self.groups], 1
        )
        layers = groups.float().cpu().numpy()  # [50 Hz, nhóm, 1024]
        if len(layers) % 2:
            layers = np.concatenate([layers, layers[-1:]])
        ssl = layers.reshape(-1, 2, *layers.shape[1:]).mean(1)
        pad = ((0, max(0, frames - len(ssl))), (0, 0), (0, 0))
        ssl = np.pad(ssl, pad, mode="edge")[:frames]
        save_npz(
            output,
            ssl=ssl.astype(np.float16),
            times_s=np.arange(frames) / 25,
            valid=np.ones(frames, bool),
        )
        meta = {
            "format": "xlsr-25hz-v2",
            "step_s": 0.04,
            "duration_s": frames / 25,
            "feature_signature": self.signature,
            "feature_sha256": sha(output),
            "source_fingerprint": source,
        }
        write_json(Path(output).with_suffix(".json"), meta)
        return meta


def native_frames(video, count, color=False):
    """Frame xám (hoặc BGR) ở độ phân giải gốc trên lưới 25 Hz (đúng count frame)."""
    import av
    import cv2

    frames, grid, origin = [], 0.0, None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            t = float(frame.pts * frame.time_base)
            origin = t if origin is None else origin
            if t - origin < grid - 0.02:
                continue
            bgr = frame.to_ndarray(format="bgr24")
            frames.append(bgr if color else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            grid += 0.04
            if len(frames) == count:
                break
    while frames and len(frames) < count:
        frames.append(frames[-1])
    return frames


def crop_square(gray, cx, cy, side, size):
    import cv2

    scale = size / side
    matrix = np.array([[scale, 0, size / 2 - scale * cx], [0, scale, size / 2 - scale * cy]])
    return cv2.warpAffine(
        gray,
        matrix,
        (size, size),
        flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def normalized_lips(landmarks):
    """[T,68,2] → [T,20,2]: môi 48-67, trừ tâm miệng, chia bề rộng hai mép (48-54)."""
    lips = landmarks[:, 48:68].astype(np.float32)
    center = lips.mean(1, keepdims=True)
    width = np.linalg.norm(lips[:, 6] - lips[:, 0], axis=1)[:, None, None]
    return ((lips - center) / np.maximum(width, 1e-3)).astype(np.float16)


# Mô hình 3D trung bình (mm) cho 6 landmark: mũi 30, cằm 8, khoé mắt 36 / 45, khoé miệng 48 / 54.
FACE_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ]
)
FACE_POINTS = (30, 8, 36, 45, 48, 54)


def head_pose(landmarks, valid, side=640.0):
    """[T,68,2] → [T,3] yaw / pitch / roll (độ / 90) bằng solvePnP; frame không có mặt = 0.

    Dùng cho luồng "lệch môi–đầu": lip-sync vẽ lại môi theo tiếng mới nhưng giữ tư thế đầu của
    câu gốc (LipDA, ICML 2026). Camera giả định: tiêu cự = cạnh dài khung, tâm ở giữa khung.
    """
    import cv2

    camera = np.array([[side, 0, side / 2], [0, side, side / 2], [0, 0, 1]], float)
    out = np.zeros((len(landmarks), 3), np.float32)
    for i in np.flatnonzero(valid):
        points = landmarks[i, list(FACE_POINTS)].astype(float)
        ok, rvec, _ = cv2.solvePnP(FACE_3D, points, camera, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        pitch = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = np.degrees(np.arcsin(-np.clip(rotation[2, 0], -1, 1)))
        roll = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
        pitch = (pitch + 180) % 360 - 180
        pitch = pitch - 180 if pitch > 90 else pitch + 180 if pitch < -90 else pitch
        out[i] = np.array([yaw, pitch, roll]) / 90.0
    return out


class MouthEncoder:
    kind = "mouth"

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.size = int(cfg.get("crop", 96))
        self.signature = fingerprint(
            {
                "code": sha(__file__),
                "crop": self.size,
                "box_max_side": cfg.get("box_max_side", 640),
                "format": "mouth-native-v2",
            }
        )

    def extract(self, request, output):
        video = Path(request["video"])
        source = source_fingerprint(video)
        box_path = Path(request["box_cache"])
        box_meta = read_json(box_path.with_suffix(".json"))
        signature = fingerprint([self.signature, box_meta["feature_sha256"]])
        if fresh(output, signature, source):
            return read_json(Path(output).with_suffix(".json"))
        with np.load(box_path, allow_pickle=False) as z:
            boxes = z["mouth_boxes"].astype(np.float32)
            landmarks = z["landmarks"].astype(np.float32)
            valid = (z["visual_valid"] & (boxes[:, 2] > 0)).copy()
        frames = native_frames(video, len(boxes))
        if not frames:
            raise ValueError(f"No frames decoded: {video}")
        height, width = frames[0].shape
        scale = max(height, width) / min(max(height, width), self.cfg.get("box_max_side", 640))
        mouth = np.zeros((len(boxes), self.size, self.size), np.uint8)
        for i, (cx, cy, side) in enumerate(boxes):
            if valid[i]:
                mouth[i] = crop_square(frames[i], cx * scale, cy * scale, side * scale, self.size)
        save_npz(
            output,
            mouth=mouth,
            lips=normalized_lips(landmarks),
            pose=head_pose(landmarks, valid, float(self.cfg.get("box_max_side", 640))),
            valid=valid,
            times_s=np.arange(len(boxes)) / 25,
            native_side=np.float32(boxes[:, 2].mean() * scale),
        )
        meta = {
            "format": "mouth-native-v2",
            "step_s": 0.04,
            "duration_s": len(boxes) / 25,
            "feature_signature": signature,
            "feature_sha256": sha(output),
            "source_fingerprint": source,
            "box_sha256": box_meta["feature_sha256"],
        }
        write_json(Path(output).with_suffix(".json"), meta)
        return meta


# ---------------------------------------------------------------- LIPINC-V2 (Datta et al. 2025)
LIPINC_FACE = 256  # FaceAligner(desiredFaceWidth=256, desiredLeftEye=(0.40, 0.40)) của bản gốc
LIPINC_CROP = (144, 64)  # cv2.resize(lipcrop, (144, 64)) → ảnh 64 × 144
LIPINC_LOCAL, LIPINC_GLOBAL = 5, 3


def lipinc_align(points):
    """Ma trận affine căn mặt đúng như `FaceAligner.align` của LIPINC-V2 (imutils)."""
    import cv2

    left = points[42:48].mean(0).astype(int)  # FACIAL_LANDMARKS_IDXS["left_eye"] = (42, 48)
    right = points[36:42].mean(0).astype(int)
    dy, dx = right[1] - left[1], right[0] - left[0]
    angle = np.degrees(np.arctan2(dy, dx)) - 180
    desired = (1.0 - 0.40 - 0.40) * LIPINC_FACE
    scale = desired / max(np.hypot(dx, dy), 1e-6)
    center = (int((left[0] + right[0]) // 2), int((left[1] + right[1]) // 2))
    matrix = cv2.getRotationMatrix2D(center, float(angle), float(scale))
    matrix[0, 2] += LIPINC_FACE * 0.5 - center[0]
    matrix[1, 2] += LIPINC_FACE * 0.40 - center[1]
    return matrix


def lipinc_ratios(points):
    """Bốn tỉ lệ dùng để chọn frame (miệng mở, miệng rộng, hai má) trên landmark đã căn.

    Bản gốc chia cho hộp mặt của bộ dò dlib chạy lại trên mặt đã căn; ở đây hộp mặt lấy từ chính
    landmark (hàm 0-16, lông mày 17-26) vì cache không giữ hộp dò. Mặt đã căn cùng cỡ nên hai cách
    chỉ lệch nhau gần như một hằng số (khác biệt ghi trong docs/v003_model.md).
    """
    width = max(points[0:17, 0].max() - points[0:17, 0].min(), 1e-6)
    height = max(points[0:17, 1].max() - points[17:27, 1].min(), 1e-6)

    def dist(a, b):
        return float(np.hypot(*(points[a] - points[b])))

    return (dist(62, 66) / height, dist(60, 64) / width, dist(48, 4) / width, dist(54, 12) / width)


def lipinc_select(ratios):
    """Chọn 5 frame cục bộ + 3 frame toàn cục theo `find_LGframes` / `find_global_frames`.

    ratios: danh sách (vị trí frame, 4 tỉ lệ) của các frame có mặt, đúng thứ tự `face_array` gốc.
    Trả (local, global, fallback) là chỉ số trong `ratios`; fallback=True khi luật gốc không tìm
    được bộ nào (bản gốc báo lỗi "no_face" và bỏ video; ở đây vẫn chọn để có điểm cho mọi sample).
    """
    local = []
    for k, (_, (lh, lw, lch, rch)) in enumerate(ratios):
        if lh * 100 < 1:  # miệng khép: bắt đầu lại chuỗi frame liên tiếp
            local = []
            continue
        local.append(k)
        if len(local) < LIPINC_LOCAL:
            continue
        found, j = [], -1
        target = [int(v * 100) for v in (lh, lw, lch, rch)]
        while j < len(ratios) - 1:
            j += 1
            if j == k:
                continue
            cand = [int(v * 100) for v in ratios[j][1]]
            if all(c - 3 <= t < c + 3 for t, c in zip(target, cand)):  # range(c - 3, c + 3)
                found.append(j)
                j += 3
                if len(found) == LIPINC_GLOBAL:
                    return local, found, False
        local = []
    # Không có bộ nào theo luật gốc: 5 frame liên tiếp mở miệng nhất + 3 frame gần tư thế nhất.
    n = len(ratios)
    if n < LIPINC_LOCAL + LIPINC_GLOBAL:
        return [], [], True
    opening = np.array([r[0] for _, r in ratios])
    start = int(np.argmax(np.convolve(opening, np.ones(LIPINC_LOCAL), "valid")))
    local = list(range(start, start + LIPINC_LOCAL))
    ref = np.array(ratios[local[-1]][1])
    order = np.argsort([np.abs(np.array(r) - ref).sum() for _, r in ratios])
    found = []
    for spaced in (True, False):  # ưu tiên cách nhau > 3 frame như luật gốc, thiếu thì lấp
        for j in map(int, order):
            if len(found) == LIPINC_GLOBAL:
                break
            if j in local or j in found or (spaced and any(abs(j - f) <= 3 for f in found)):
                continue
            found.append(j)
    return local, found, True


class LipincEncoder:
    """8 crop miệng màu (BGR, uint8) 64 × 144 theo đúng cách tiền xử lý của LIPINC-V2.

    Landmark lấy từ cache AV-HuBERT (dlib 68 điểm, cùng bộ dự đoán với bản gốc) thay vì chạy lại
    dlib trên mặt đã căn: căn mặt bằng cùng ma trận affine rồi biến đổi landmark theo ma trận đó.
    """

    kind = "lipinc"

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.signature = fingerprint(
            {
                "code": sha(__file__),
                "box_max_side": cfg.get("box_max_side", 640),
                "format": "lipinc-v1",
            }
        )

    def extract(self, request, output):
        import cv2

        video = Path(request["video"])
        source = source_fingerprint(video)
        box_path = Path(request["box_cache"])
        box_meta = read_json(box_path.with_suffix(".json"))
        signature = fingerprint([self.signature, box_meta["feature_sha256"]])
        if fresh(output, signature, source):
            return read_json(Path(output).with_suffix(".json"))
        with np.load(box_path, allow_pickle=False) as z:
            landmarks = z["landmarks"].astype(np.float32)
            valid = z["visual_valid"].copy()
        frames = native_frames(video, len(landmarks), color=True)
        if not frames:
            raise ValueError(f"No frames decoded: {video}")
        height, width = frames[0].shape[:2]
        scale = max(height, width) / min(max(height, width), self.cfg.get("box_max_side", 640))
        aligned, ratios = {}, []
        for i in np.flatnonzero(valid[: len(frames)]):
            points = landmarks[i] * scale
            matrix = lipinc_align(points)
            moved = cv2.transform(points[None], matrix)[0]
            aligned[int(i)] = (matrix, moved)
            ratios.append((int(i), lipinc_ratios(moved)))
        local, found, fallback = lipinc_select(ratios)
        chosen = [ratios[k][0] for k in local + found]
        slots = LIPINC_LOCAL + LIPINC_GLOBAL
        crops = np.zeros((slots, LIPINC_CROP[1], LIPINC_CROP[0], 3), np.uint8)
        for slot, i in enumerate(chosen):
            matrix, moved = aligned[i]
            face = cv2.warpAffine(
                frames[i], matrix, (LIPINC_FACE, LIPINC_FACE), flags=cv2.INTER_CUBIC
            )
            x, y, w, h = cv2.boundingRect(np.round(moved[48:68]).astype(np.int32))
            x, y = max(0, x), max(0, y)
            lip = face[y : y + max(h, 1), x : x + max(w, 1)]
            if lip.size:
                crops[slot] = cv2.resize(lip, LIPINC_CROP)
        save_npz(
            output,
            frames=crops,
            frame_ids=np.array(chosen + [-1] * (slots - len(chosen)), np.int32),
            valid=valid,
            times_s=np.arange(len(landmarks)) / 25,
        )
        meta = {
            "format": "lipinc-v1",
            "step_s": 0.04,
            "duration_s": len(landmarks) / 25,
            "selected": len(chosen),
            "fallback": bool(fallback),
            "feature_signature": signature,
            "feature_sha256": sha(output),
            "source_fingerprint": source,
            "box_sha256": box_meta["feature_sha256"],
        }
        write_json(Path(output).with_suffix(".json"), meta)
        return meta


# ---------------------------------------------------------------- waveform cho XLS-R fine-tune
class WaveEncoder:
    """Tiếng 16 kHz mono, không chuẩn hoá (như SSL_Anti-spoofing), xếp [T, 640] int16."""

    kind = "wave"

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.signature = fingerprint({"code": sha(__file__), "format": "wave-16k-v1"})

    def extract(self, request, output):
        video = Path(request["video"])
        source = source_fingerprint(video)
        if fresh(output, self.signature, source):
            return read_json(Path(output).with_suffix(".json"))
        pcm = audio_16k(video)
        frames = max(1, int(round(len(pcm) / 640)))
        pcm = np.pad(pcm, (0, max(0, frames * 640 - len(pcm))))[: frames * 640]
        wave = np.clip(np.round(pcm * 32767), -32768, 32767).astype(np.int16)
        save_npz(
            output,
            wave=wave.reshape(frames, 640),
            valid=np.ones(frames, bool),
            times_s=np.arange(frames) / 25,
        )
        meta = {
            "format": "wave-16k-v1",
            "step_s": 0.04,
            "duration_s": frames / 25,
            "feature_signature": self.signature,
            "feature_sha256": sha(output),
            "source_fingerprint": source,
        }
        write_json(Path(output).with_suffix(".json"), meta)
        return meta
