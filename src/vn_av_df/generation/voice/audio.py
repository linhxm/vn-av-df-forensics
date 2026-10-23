"""Xử lý tiếng dùng chung cho ② sham và ③ / ⑥ giọng AI (cùng một cách ghép, để vết ghép
không trở thành dấu hiệu riêng của giọng AI).

Mọi PCM là float32 mono 48 kHz, cùng lưới thời gian với video (1920 mẫu / frame 25 Hz).
"""

import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np

RATE = 48000


def resample(pcm, rate, target=RATE):
    from scipy.signal import resample_poly

    pcm = np.asarray(pcm, np.float32)
    if rate == target:
        return pcm
    g = gcd(int(rate), int(target))
    return resample_poly(pcm, target // g, int(rate) // g).astype(np.float32)


def read_wav(path, target=RATE):
    from scipy.io import wavfile

    rate, data = wavfile.read(str(path))
    data = np.asarray(data)
    if data.ndim == 2:
        data = data.mean(1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return resample(data.astype(np.float32), rate, target)


def write_wav(path, pcm, rate=RATE):
    from scipy.io import wavfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.wav")
    wavfile.write(partial, rate, np.clip(np.asarray(pcm) * 32767, -32768, 32767).astype(np.int16))
    partial.replace(path)


def rms(pcm):
    pcm = np.asarray(pcm, np.float64)
    return float(np.sqrt(np.mean(pcm**2))) if len(pcm) else 0.0


def trim_silence(pcm, threshold_db=-40.0, frame=480):
    """Bỏ lặng đầu/cuối của câu TTS (theo năng lượng 10 ms) trước khi co giãn vào chỗ trống."""
    pcm = np.asarray(pcm, np.float32)
    if len(pcm) < frame:
        return pcm
    blocks = pcm[: len(pcm) // frame * frame].reshape(-1, frame)
    level = 20 * np.log10(np.sqrt((blocks**2).mean(1)) + 1e-9)
    loud = np.flatnonzero(level > level.max() + threshold_db)
    if not len(loud):
        return pcm
    return pcm[loud[0] * frame : (loud[-1] + 1) * frame]


def time_stretch(pcm, target_samples, ffmpeg):
    """Đổi độ dài không đổi cao độ bằng bộ lọc atempo của ffmpeg; trả (pcm, tỉ lệ)."""
    pcm = np.asarray(pcm, np.float32)
    if not len(pcm) or target_samples <= 0:
        raise ValueError("Empty audio cannot be stretched")
    ratio = len(pcm) / target_samples  # >1: phải nói nhanh hơn
    if abs(ratio - 1) < 0.005:
        out = pcm
    else:
        chain, left = [], ratio
        while left > 2.0:
            chain.append(2.0)
            left /= 2.0
        while left < 0.5:
            chain.append(0.5)
            left /= 0.5
        chain.append(left)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
            write_wav(src, pcm)
            subprocess.run(
                [
                    str(ffmpeg),
                    *("-nostdin", "-v", "error", "-y", "-i", str(src)),
                    *("-af", ",".join(f"atempo={c:.6f}" for c in chain), str(dst)),
                ],
                check=True,
                capture_output=True,
            )
            out = read_wav(dst)
    out = out[:target_samples]
    if len(out) < target_samples:
        out = np.pad(out, (0, target_samples - len(out)))
    return out.astype(np.float32), float(ratio)


def noise_floor(pcm, seconds=0.2):
    """Đoạn yên nhất (theo RMS 200 ms) của tiếng gốc, dùng làm tiếng nền phủ lên đoạn chèn."""
    size = int(seconds * RATE)
    pcm = np.asarray(pcm, np.float32)
    if len(pcm) < 2 * size:
        return np.zeros(size, np.float32)
    starts = range(0, len(pcm) - size, size // 2)
    best = min(starts, key=lambda s: rms(pcm[s : s + size]))
    return pcm[best : best + size].copy()


def frame_db(pcm, frame=960):
    """Mức (dB) từng khung 20 ms."""
    pcm = np.asarray(pcm, np.float64)
    blocks = pcm[: len(pcm) // frame * frame].reshape(-1, frame)
    return 20 * np.log10(np.sqrt((blocks**2).mean(1)) + 1e-9)


def edge_silence(pcm, frame=960, floor_db=30.0):
    """(lặng đầu, lặng cuối) tính bằng mẫu: khung thấp hơn khung to nhất floor_db dB là lặng
    (cùng định nghĩa với audit.py)."""
    db = frame_db(pcm, frame)
    loud = db > db.max() - floor_db if len(db) else np.zeros(0, bool)
    if not loud.any():
        return len(pcm), 0
    first = int(np.argmax(loud)) * frame
    return first, len(pcm) - (len(loud) - int(np.argmax(loud[::-1]))) * frame


def fit_error(window, samples, lead, tail):
    """Độ lệch (giây) giữa đoạn nói fit_donor chọn và chỗ trống samples - lead - tail."""
    return abs(window[1] - window[0] - (samples - lead - tail) / RATE)


def crossfade_join(parts, fade):
    """Nối các đoạn, chồng `fade` mẫu ở mỗi chỗ nối (đoạn ngắn hơn thì chồng ít hơn)."""
    out = np.zeros(0, np.float32)
    for part in parts:
        part = np.asarray(part, np.float32)
        f = min(fade, len(out), len(part))
        if f:
            ramp = np.linspace(0, 1, f, dtype=np.float32)
            part = part.copy()
            part[:f] = part[:f] * ramp + out[-f:] * (1 - ramp)
            out = out[:-f]
        out = np.concatenate([out, part])
    return out


def fit_donor(donor, samples, lead, tail, min_pause_s=0.02, floor_db=30.0, frame=960):
    """Tiếng clip cho tiếng (② sham, ⑤) dài đúng `samples`, lặng đầu / cuối đúng như clip gốc.

    Cắt cứng donor[:samples] (cách cũ) hết tiếng giữa câu: ở part 1, 79 % sham full không có
    khoảng lặng cuối, real chỉ 3 % (AUC khoảng lặng cuối 0,08). Ở đây lấy đoạn nói donor[o:e] dài
    gần samples - lead - tail nhất: o bắt đầu sau một chỗ nghỉ (≥ min_pause_s lặng) hoặc đầu clip,
    e dừng trước một chỗ nghỉ hoặc cuối clip. Hai đầu dài đúng lead / tail: tiếng lặng thật sát
    đoạn nói, thiếu thì tiếng nền của chính donor. Phần dư kéo dài chỗ nghỉ dài nhất trong đoạn nói
    (không có thì dồn vào cuối). Lặng tính so với khung to nhất của đoạn nói (có thể nhỏ hơn của
    cả donor); tiếng nền được hạ xuống dưới ngưỡng đó.
    Trả (pcm, cửa sổ [o, e] giây trong donor), hoặc (None, None) nếu không đoạn nào vừa;
    `fit_error(window, ...)` đo độ lệch để chọn donor vừa nhất.
    """
    donor = np.asarray(donor, np.float32)
    db = frame_db(donor, frame)
    n = len(db)
    if not n:
        return None, None
    need = max(1, round(min_pause_s * RATE / frame))

    def pauses(loud):
        edges = np.flatnonzero(np.diff(np.r_[False, ~loud, False].astype(np.int8)))
        return [(int(s), int(t)) for s, t in zip(edges[::2], edges[1::2])]

    loud = db > db.max() - floor_db
    runs = [(s, t) for s, t in pauses(loud) if t - s >= need or s == 0 or t == n]
    onsets = sorted({t for s, t in runs if t < n} | ({0} if loud[0] else set()))
    offsets = sorted({s for s, t in runs if s > 0})
    # Đoạn nói gần đích nhất về cả hai phía (dài hơn thì lặng cuối ngắn lại, không âm): sai số
    # đối xứng quanh clip gốc, không lệch một chiều.
    target, room = (samples - lead - tail) / frame, (samples - lead) // frame
    pairs = [(o, e) for o in onsets for e in offsets if o < e <= o + room]
    if not pairs:
        return None, None
    o, e = min(pairs, key=lambda p: (abs(p[1] - p[0] - target), -(p[1] - p[0])))
    tail = min(tail, samples - lead - (e - o) * frame)
    ceiling = db[o:e].max() - floor_db - 3  # 3 dB dự phòng: vẫn lặng sau khi ghép
    quiet = db < ceiling
    start, stop = o, e
    while start > 0 and quiet[start - 1] and (o - start) * frame < lead:
        start -= 1
    while stop < n and quiet[stop] and (stop - e) * frame < tail:
        stop += 1
    noise = noise_floor(donor)
    if len(noise) >= frame:
        noise = noise * min(1.0, 10 ** ((ceiling - 3 - frame_db(noise, frame).max()) / 20))

    def fill(length):
        if length <= 0:
            return np.zeros(0, np.float32)
        if not len(noise):
            return np.zeros(length, np.float32)
        return np.tile(noise, length // len(noise) + 1)[:length]

    fade = int(0.005 * RATE)
    extra = samples - lead - tail - (e - o) * frame
    inner = [(s, t) for s, t in pauses(~quiet[o:e]) if 0 < s and t < e - o and t - s >= need]
    if inner:
        s, t = max(inner, key=lambda r: r[1] - r[0])
        m = (o + (s + t) // 2) * frame
        speech = [donor[o * frame : m], fill(extra + 2 * fade), donor[m : e * frame]]
        extra = 0
    else:
        speech = [donor[o * frame : e * frame]]
    pre = donor[max(start * frame, o * frame - lead) : o * frame]
    post = donor[e * frame : min(stop * frame, e * frame + tail)]
    parts = [
        fill(lead - len(pre) + fade),
        pre,
        *speech,
        post,
        fill(tail - len(post) + extra + fade),
    ]
    out = crossfade_join([p for p in parts if len(p)], fade)
    out = np.concatenate([out, fill(samples - len(out))])[:samples]
    return out.astype(np.float32), [round(o * frame / RATE, 3), round(e * frame / RATE, 3)]


def splice(base, insert, start, end, crossfade_s=0.015, match_loudness=True, noise=None):
    """Thay base[start:end] (mẫu) bằng insert (đã đúng độ dài), crossfade hai đầu.

    match_loudness: đưa RMS đoạn chèn về RMS đoạn bị thay. noise: tiếng nền lặp lại để phủ
    đoạn chèn (TTS sạch hơn hẳn tiếng thu thật).
    """
    base = np.asarray(base, np.float32).copy()
    insert = np.asarray(insert, np.float32).copy()
    if len(insert) != end - start or not 0 <= start < end <= len(base):
        raise ValueError("Insert must exactly fill a valid span")
    if match_loudness:
        target, current = rms(base[start:end]), rms(insert)
        if current > 1e-6 and target > 1e-6:
            insert *= target / current
    if noise is not None and len(noise):
        reps = int(np.ceil(len(insert) / len(noise)))
        insert += np.tile(noise, reps)[: len(insert)]
    fade = max(1, min(int(crossfade_s * RATE), (end - start) // 4))
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    insert[:fade] = insert[:fade] * ramp + base[start : start + fade] * (1 - ramp)
    insert[-fade:] = insert[-fade:] * ramp[::-1] + base[end - fade : end] * (1 - ramp[::-1])
    base[start:end] = insert
    return np.clip(base, -1, 1)


def fit_whole(pcm, samples, ffmpeg, max_ratio=0.15):
    """Đưa cả câu TTS về đúng độ dài clip: co giãn trong ±max_ratio, phần dư đệm lặng / cắt."""
    pcm = trim_silence(pcm)
    ratio = len(pcm) / samples
    target = int(round(len(pcm) / min(max(ratio, 1 - max_ratio), 1 + max_ratio)))
    out, _ = time_stretch(pcm, target, ffmpeg)
    if len(out) >= samples:
        return out[:samples], ratio
    pad = samples - len(out)
    return np.pad(out, (pad // 2, pad - pad // 2)), ratio
