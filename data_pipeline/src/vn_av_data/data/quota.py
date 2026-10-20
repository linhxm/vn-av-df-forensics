"""Luật số lượng của bước cắt: lấy bao nhiêu clip ứng viên và thử cửa sổ theo thứ tự nào.

Tách khỏi `cut.py` vì không đổi cách làm ra một clip (độ dài, kiểm mặt, chuyển cảnh, jump cut,
VAD, độ lệch): file này không nằm trong cut signature, nên đổi chỉ tiêu không chặn batch đã cắt.
Mỗi video ghi luật đã dùng vào `cut_info` (`mode`, `quota`).

- `1` (một người): 30-40 clip cho cả nguồn, chia theo lượng tiếng nói giữa các video.
- `2` (hai người): tối đa `cut.pair.max_clips` clip mỗi video (mặc định 150, ~75 / người).
- `n` (nhiều người): không giới hạn, cắt mọi cửa sổ đạt.
`1` và `2` thử cửa sổ rải đều theo thời gian (spread_order) đến khi đủ chỉ tiêu.
"""

import numpy as np


def single_target(speech_s, rules):
    """`min_clips` khi ≤ low phút tiếng nói, tăng tuyến tính tới `max_clips` ở full phút."""
    minutes = speech_s / 60
    lo, hi = rules["low_speech_minutes"], rules["full_speech_minutes"]
    share = min(1.0, max(0.0, (minutes - lo) / max(1e-6, hi - lo)))
    return int(round(rules["min_clips"] + share * (rules["max_clips"] - rules["min_clips"])))


def split_quota(total, weights):
    """Chia `total` theo trọng số, phần dư theo phần thập phân lớn nhất."""
    weights = np.asarray(weights, float)
    if total <= 0 or weights.sum() <= 0:
        return [0] * len(weights)
    raw = total * weights / weights.sum()
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: total - base.sum()]:
        base[i] += 1
    return base.tolist()


def spread_order(windows, quota, duration):
    """Thứ tự thử cửa sổ để clip rải đều: chia trục thời gian thành `quota` khúc, mỗi vòng lấy
    cửa sổ gần tâm khúc nhất còn lại của từng khúc chưa có clip."""
    if not windows or quota <= 0:
        return []
    width = duration / quota
    queues = [[] for _ in range(quota)]
    for w in windows:
        mid = (w[0] + w[1]) / 2
        queues[min(quota - 1, int(mid // width))].append(w)
    for k, q in enumerate(queues):
        center = (k + 0.5) * width
        q.sort(key=lambda w: abs((w[0] + w[1]) / 2 - center))
    return queues


def take_spread(windows, quota, duration, budget, try_window):
    """Thử cửa sổ theo spread_order đến khi được `quota` clip hoặc đã thử `budget` cửa sổ.

    Mỗi vòng: khúc chưa có clip trước; hết khúc trống thì lấy khúc còn cửa sổ (khúc không có
    clip tốt thì bù bằng khúc khác). try_window(start, end) → True nếu cửa sổ thành clip.
    Trả số cửa sổ đã thử.
    """
    queues = spread_order(windows, quota, duration)
    taken = tried = 0
    filled = set()
    while taken < quota and tried < budget and any(queues):
        live = [k for k, q in enumerate(queues) if q]
        for k in [k for k in live if k not in filled] or live:
            if taken >= quota or tried >= budget:
                break
            start, end = queues[k].pop(0)
            tried += 1
            if try_window(start, end):
                taken += 1
                filled.add(k)
    return tried
