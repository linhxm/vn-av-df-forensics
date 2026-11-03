"""Ngữ điệu thị giác (visual prosody) cho M3: đặc trưng ngữ điệu của tiếng và chuyển động đầu / lông
mày theo frame 25 Hz, cùng phép kiểm trước khi train.

Căn cứ (RESEARCH_WORKLOG/v004_model.md §6): chuyển động đầu gắn với F0 và năng lượng giọng (Yehia,
Kuratate, Vatikiotis-Bateson, J. Phonetics 2002; Deng et al. 2004; Busso et al. 2005); ở tiếng có
thanh điệu, chuyển động đầu / lông mày tham gia tạo thanh (Burnham et al., Speech Communication
2024 với tiếng Quảng Đông; Garg et al. 2019 với tiếng Quan Thoại; tiếng Thái, AVSP 2019). Generator
lip-sync chỉ vẽ lại miệng: khi tiếng bị thay (② sham, ③ voice, ⑤ donor, ⑥ một cụm), đầu vẫn chuyển
động theo câu gốc nên cặp "ngữ điệu ↔ đầu" lệch, trong khi ở ⑤ cặp "tiếng ↔ môi" lại khớp.

Đặc trưng tiếng (`prosody`, [T, 5]): log F0 chuẩn hoá trong clip (0 ở frame vô thanh), xác suất hữu
thanh, log năng lượng chuẩn hoá, sai phân của hai đại lượng. F0 bằng tự tương quan chuẩn hoá trên
cửa sổ 64 ms quanh mỗi frame, dải 70-400 Hz (giọng nói người lớn).
Đặc trưng đầu (`head`, [T, 5]): độ nâng lông mày (khoảng cách lông mày - mắt / khoảng cách hai mắt,
chuẩn hoá trong clip), sai phân của nó, vận tốc yaw / pitch / roll.
"""

import hashlib
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

SAMPLE_RATE, HOP = 16000, 640
WINDOW = 1024  # 64 ms
F0_RANGE = (70.0, 400.0)
VOICED_THRESHOLD = 0.45


def _zscore(x, keep=None):
    keep = np.ones(len(x), bool) if keep is None else keep
    if keep.sum() < 2:
        return np.zeros_like(x)
    mean, std = x[keep].mean(), x[keep].std() + 1e-6
    out = (x - mean) / std
    out[~keep] = 0.0
    return out


def _delta(x):
    return np.concatenate([np.zeros_like(x[:1]), np.diff(x, axis=0)])


def prosody_features(wave):
    """wave [T, 640] (16 kHz, đã chia 32768) → [T, 5] float32."""
    wave = np.asarray(wave, np.float32)
    frames = len(wave)
    if frames == 0:
        return np.zeros((0, 5), np.float32)
    flat = wave.reshape(-1)
    pad = WINDOW // 2
    padded = np.pad(flat, (pad, pad + WINDOW))
    starts = np.arange(frames) * HOP + HOP // 2  # tâm frame (trên tín hiệu đã đệm)
    index = starts[:, None] + np.arange(WINDOW)[None]
    x = padded[index] * np.hanning(WINDOW)[None].astype(np.float32)
    x = x - x.mean(1, keepdims=True)
    energy = (x**2).mean(1)
    spectrum = np.fft.rfft(x, n=2 * WINDOW)
    acf = np.fft.irfft(np.abs(spectrum) ** 2)[:, :WINDOW]
    acf = acf / np.maximum(acf[:, :1], 1e-9)
    low = int(SAMPLE_RATE / F0_RANGE[1])
    high = int(SAMPLE_RATE / F0_RANGE[0])
    lag = low + np.argmax(acf[:, low:high], axis=1)
    peak = acf[np.arange(frames), lag]
    log_energy = np.log(energy + 1e-8)
    loud = log_energy > np.percentile(log_energy, 20)
    voiced = (peak > VOICED_THRESHOLD) & loud
    log_f0 = np.log(SAMPLE_RATE / lag.astype(np.float32))
    f0 = _zscore(log_f0, voiced)
    e = _zscore(log_energy)
    voicing = np.clip(peak, 0, 1) * loud
    out = np.stack([f0, voicing, e, _delta(f0) * voiced, _delta(e)], 1)
    return out.astype(np.float32)


EYEBROWS, EYES = slice(17, 27), slice(36, 48)


def head_features(landmarks, pose, valid):
    """landmarks [T, 68, 2], pose [T, 3] (yaw / pitch / roll), valid [T] → [T, 5] float32."""
    landmarks = np.asarray(landmarks, np.float32)
    pose = np.asarray(pose, np.float32)
    valid = np.asarray(valid, bool)
    n = min(len(landmarks), len(pose), len(valid))
    landmarks, pose, valid = landmarks[:n], pose[:n], valid[:n]
    if n == 0:
        return np.zeros((0, 5), np.float32)
    brow = landmarks[:, EYEBROWS, 1].mean(1)
    eye = landmarks[:, EYES, 1].mean(1)
    left, right = landmarks[:, 36:42].mean(1), landmarks[:, 42:48].mean(1)
    scale = np.linalg.norm(left - right, axis=1)
    ok = valid & (scale > 1e-3)
    raise_ = np.where(ok, (eye - brow) / np.maximum(scale, 1e-3), 0.0)
    raise_ = _zscore(raise_, ok)
    velocity = _delta(pose) * ok[:, None] * 10.0  # pose đã chia 90°: vận tốc nhỏ → khuếch đại
    out = np.concatenate([raise_[:, None], (_delta(raise_) * ok)[:, None], velocity], 1)
    return out.astype(np.float32)


# ---------------------------------------------------------------- kiểm trước khi train
def coupling(prosody, head, valid, max_lag=12):
    """Tương quan lớn nhất (theo độ trễ ±max_lag) giữa "độ hoạt động ngữ điệu" (|ΔF0| + |Δ năng
    lượng|) và tốc độ chuyển động đầu / lông mày. Một số mỗi clip; NaN nếu quá ngắn."""
    n = min(len(prosody), len(head), len(valid))
    if n < 2 * max_lag + 10:
        return math.nan
    p = np.abs(prosody[:n, 3]) + np.abs(prosody[:n, 4])
    h = np.abs(head[:n, 1]) + np.linalg.norm(head[:n, 2:5], axis=1)
    keep = np.asarray(valid[:n], bool)
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        lo, hi = max(0, lag), min(n, n + lag)
        a, b = p[lo:hi], h[lo - lag : hi - lag]
        k = keep[lo:hi] & keep[lo - lag : hi - lag]
        if k.sum() < 10 or a[k].std() < 1e-6 or b[k].std() < 1e-6:
            continue
        best = max(best, float(np.corrcoef(a[k], b[k])[0, 1]))
    return best if best > -1 else math.nan


def audit_sample(rows, per_group=None):
    """Lấy tối đa per_group mẫu mỗi (split, ô, kiểu), chọn theo băm sample_id: ổn định giữa các lần
    chạy và rải đều các part (manifest xếp theo part, lấy đầu danh sách sẽ chỉ có part 1)."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row.get("cell"), row.get("kind"))].append(row)
    picked = []
    for group in groups.values():
        group.sort(key=lambda r: hashlib.sha1(r["sample_id"].encode()).hexdigest())
        picked.extend(group[:per_group])
    return picked


def clip_coupling(files):
    """{encoder: file} của một clip → coupling (số, NaN nếu quá ngắn); None nếu thiếu cache."""
    if not all(path.is_file() for path in files.values()):
        return None
    with np.load(files["wave"], allow_pickle=False) as z:
        prosody = prosody_features(z["wave"].astype(np.float32) / 32768.0)
    with np.load(files["avhubert"], allow_pickle=False) as z:
        landmarks, valid = z["landmarks"], z["visual_valid"]
    with np.load(files["mouth"], allow_pickle=False) as z:
        pose = z["pose"] if "pose" in z.files else np.zeros((len(z["lips"]), 3))
    head = head_features(landmarks, pose, valid)
    return coupling(prosody, head, valid[: len(head)])


def audit(cfg, rows, per_group=200, workers=8, every=500):
    """Ngữ điệu ↔ đầu có tách được real với các ô thay tiếng không, **trước khi train** (không có
    tham số học): AUC của −coupling (cao = lệch) cho từng ô so với real cùng split. Nếu mọi AUC
    ≈ 0,5 thì nhánh ngữ điệu khó có ích trên dữ liệu này.

    Chỉ là phép kiểm nên lấy mẫu theo nhóm: 200 + 200 mẫu cho sai số chuẩn của AUC quanh 0,5 ≈ 0,03.
    per_group=None: mọi mẫu. Đọc từ /kaggle/input rất chậm (cả 23.242 mẫu tuần tự: sau hơn 3 giờ vẫn
    chưa xong, 03/11) nên đọc song song, tra file bằng cache_index và in tiến độ."""
    from sklearn.metrics import roc_auc_score

    from vn_av_df.training.data import cache_index

    picked = audit_sample(rows, per_group)
    ids = [r["sample_id"] for r in picked]
    index = {e: cache_index(cfg, e, ids) for e in ("avhubert", "mouth", "wave")}
    print(f"Kiểm ngữ điệu–đầu: {len(picked)} / {len(rows)} mẫu", flush=True)
    values, missing, start = defaultdict(list), 0, time.time()
    with ThreadPoolExecutor(workers) as pool:  # đa phần chờ đọc đĩa mạng; zlib, FFT nhả GIL
        jobs = {
            pool.submit(clip_coupling, {e: index[e][r["sample_id"]] for e in index}): r
            for r in picked
        }
        for done, job in enumerate(as_completed(jobs), 1):
            row, score = jobs[job], job.result()
            missing += score is None
            if score is not None and math.isfinite(score):
                values[(row["split"], row.get("cell"), row.get("kind"))].append(-score)
            if done % every == 0 or done == len(picked):
                print(f"  {done} / {len(picked)} mẫu, {time.time() - start:.0f} s", flush=True)
    print(f"  thiếu cache: {missing} mẫu", flush=True)
    out = {}
    for (split, cell, kind), scores in sorted(values.items(), key=str):
        reals = [v for (s, c, _), vs in values.items() if s == split and c == "real" for v in vs]
        if cell == "real" or not reals:
            continue
        labels = [1] * len(scores) + [0] * len(reals)
        out[f"{split}/{cell}/{kind}"] = {
            "n": len(scores),
            "auc_vs_real": float(roc_auc_score(labels, scores + reals)),
        }
    return out
