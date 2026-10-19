"""Bước 06 parts: gom speaker đã duyệt đủ thành part cố định (mặc định 20 nam + 20 nữ, mỗi người
15-20 clip).

Điều kiện một speaker vào part: duyệt xong, đã gán giới, số clip được chọn trong
[min_clips_per_speaker, max_clips_per_speaker].
Speaker xuất hiện chung một video (talkshow, phỏng vấn) thành một nhóm: cả nhóm vào cùng part và
cùng split (không rò cảnh quay giữa train và test). Nhóm còn speaker chưa duyệt thì chờ.
Lấy speaker theo thứ tự duyệt xong; part tạo ra là đóng băng, phần dư nằm lại chờ part sau.
Split chia theo speaker trong part, cân giới: mặc định train 12+12, validation 3+3, test 5+5; trong
giới hạn đó, tổng số clip của mỗi split được cân theo tỉ lệ số người (người ít / nhiều clip trải đều).
"""

import random
from collections import defaultdict

from vn_av_data.ledger import now

GENDERS = ("M", "F")


def speaker_groups(ledger):
    """Nhóm liên thông speaker theo video chung (mọi clip đã gán, không chỉ clip được chọn)."""
    rows = ledger.rows(
        "SELECT DISTINCT speaker_id, video_id FROM clips WHERE speaker_id IS NOT NULL"
    )
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r["speaker_id"])
        find(r["speaker_id"])
    for speakers in by_video.values():
        for other in speakers[1:]:
            parent[find(other)] = find(speakers[0])
    groups = defaultdict(list)
    for s in parent:
        groups[find(s)].append(s)
    return [sorted(g) for g in groups.values()]


def clip_range(cfg):
    """(tối thiểu, tối đa) số clip được chọn mỗi speaker, từ mục `parts` của config."""
    lo, hi = cfg["min_clips_per_speaker"], cfg["max_clips_per_speaker"]
    if not 0 < lo <= hi:
        raise ValueError("Need 0 < min_clips_per_speaker <= max_clips_per_speaker")
    return lo, hi


def split_targets(cfg):
    return {name: dict(zip(GENDERS, value)) for name, value in cfg["split"].items()}


def assign_splits(groups, genders, targets, seed, clips=None):
    """Nhóm → split, giảm tổng bình phương lệch so với chỉ tiêu người (theo giới, ưu tiên tuyệt
    đối); phụ: tổng clip mỗi split lệch ít nhất so với tỉ lệ số người (clips: speaker → số clip)."""
    clips = clips or {}
    order = list(groups)
    random.Random(seed).shuffle(order)
    # Nhóm lớn trước, rồi nhóm nhiều clip trước (xếp kiểu "việc lớn trước" để tổng clip cân).
    order.sort(key=lambda g: (len(g), sum(clips.get(s, 0) for s in g)), reverse=True)
    people = sum(len(g) for g in groups)
    total = sum(clips.get(s, 0) for g in groups for s in g)
    share = {t: total * sum(targets[t].values()) / max(1, people) for t in targets}
    counts = {s: dict.fromkeys(GENDERS, 0) for s in targets}
    clip_sum = dict.fromkeys(targets, 0)
    result = {}
    for group in order:
        add = {g: sum(genders[s] == g for s in group) for g in GENDERS}
        n = sum(clips.get(s, 0) for s in group)

        def cost(split):
            people_cost = sum(
                (counts[t][g] + (add[g] if t == split else 0) - targets[t][g]) ** 2
                for t in targets
                for g in GENDERS
            )
            clip_cost = sum(
                ((clip_sum[t] + (n if t == split else 0) - share[t]) / max(1, total)) ** 2
                for t in targets
            )
            return (people_cost, clip_cost)

        split = min(targets, key=cost)
        for g in GENDERS:
            counts[split][g] += add[g]
        clip_sum[split] += n
        for s in group:
            result[s] = split
    return result, {t: {**counts[t], "clips": clip_sum[t]} for t in targets}


def form_parts(ledger, cfg):
    cfg = cfg["parts"]
    per_gender = cfg["speakers_per_gender"]
    lo, hi = clip_range(cfg)
    targets = split_targets(cfg)
    if any(sum(t[g] for t in targets.values()) != per_gender for g in GENDERS):
        raise ValueError("Split targets must add up to speakers_per_gender for each gender")
    speakers = {s["speaker_id"]: s for s in ledger.rows("SELECT * FROM speakers")}
    keep = {
        r["speaker_id"]: r["n"]
        for r in ledger.rows(
            "SELECT speaker_id, COUNT(*) AS n FROM clips WHERE decision='keep' GROUP BY speaker_id"
        )
    }

    def ready(s):
        return (
            s["review"] == "done"
            and not s["merged_into"]
            and s["gender"] in GENDERS
            and s["part"] is None
            and lo <= keep.get(s["speaker_id"], 0) <= hi
        )

    waiting, blocked, candidates = 0, [], []
    for group in speaker_groups(ledger):
        members = [speakers[s] for s in group if s in speakers]
        members = [m for m in members if not m["merged_into"] and m["review"] != "excluded"]
        if not members or all(m["part"] is not None for m in members):
            continue
        if any(m["part"] is not None for m in members):
            blocked.append(group)  # nối với speaker đã nằm trong part cũ: cần xử lý tay
            continue
        if not all(ready(m) for m in members):
            waiting += sum(ready(m) for m in members)
            continue
        first = min(m["reviewed_at"] or "" for m in members)
        candidates.append((first, [m["speaker_id"] for m in members]))
    candidates.sort()
    genders = {s: speakers[s]["gender"] for _, g in candidates for s in g}
    created = []
    while True:
        chosen, counts = [], dict.fromkeys(GENDERS, 0)
        for item in candidates:
            add = {g: sum(genders[s] == g for s in item[1]) for g in GENDERS}
            if all(counts[g] + add[g] <= per_gender for g in GENDERS):
                chosen.append(item)
                for g in GENDERS:
                    counts[g] += add[g]
            if all(counts[g] == per_gender for g in GENDERS):
                break
        if not all(counts[g] == per_gender for g in GENDERS):
            break
        number = (ledger.scalar("SELECT MAX(part) FROM parts") or 0) + 1
        groups = [g for _, g in chosen]
        split_of, split_counts = assign_splits(
            groups, genders, targets, seed=cfg["seed"] + number, clips=keep
        )
        members = [s for g in groups for s in g]
        with ledger.tx() as conn:
            ledger.insert(
                "parts",
                {
                    "part": number,
                    "created_at": now(),
                    "members": members,
                    "clips": sum(keep[s] for s in members),
                },
                conn=conn,
            )
            for s in members:
                ledger.update("speakers", "speaker_id", s, {"part": number}, conn)
                conn.execute(
                    "UPDATE clips SET part=?, split=? WHERE speaker_id=? AND decision='keep'",
                    (number, split_of[s], s),
                )
            ledger.log("parts", f"part {number}: {len(members)} speakers {split_counts}", conn)
        created.append({"part": number, "speakers": len(members), "splits": split_counts})
        candidates = [c for c in candidates if c not in chosen]
    left = dict.fromkeys(GENDERS, 0)
    for _, group in candidates:
        for s in group:
            left[genders[s]] += 1
    return {
        "created": created,
        "ready_left_over": left,
        "ready_but_waiting_for_group": waiting,
        "blocked_groups": blocked,
    }
