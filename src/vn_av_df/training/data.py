"""Nạp cache v003 thành tensor theo sample (lưới 25 Hz) và gom batch có đệm + mask.

Nhãn theo frame: visual / audio / mismatch = tỉ lệ frame nằm trong khoảng tương ứng; nhãn video:
label (bị AI), synthetic_visual, synthetic_audio, có mismatch.
Đặc trưng (`needs`): av (AV-HuBERT a / v), ssl (XLS-R theo nhóm lớp [T, nhóm, 1024]), mouth (crop
miệng + landmark môi + tư thế đầu), pose (chỉ landmark môi + tư thế đầu, không nạp crop), lipinc
(8 crop màu LIPINC-V2, không theo frame), wave (tiếng 16 kHz [T, 640]).
Augment độ lệch tiếng-hình (chỉ khi train): dịch đặc trưng tiếng (AV-HuBERT audio, XLS-R) so với
hình k frame, k ngẫu nhiên đều trong cfg["training"]["audio_lag_aug"] (mặc định −1…+2; dương =
tiếng trễ), như nhau cho real và fake, nhãn giữ nguyên; frame mất tiếng ở mép bị mask. Real đã
được sửa độ lệch về 0 ở data pipeline nhưng còn sai số ±1 frame và mỗi generator có "điểm 0" riêng;
khoảng −1…+2 nằm trong vùng người xem không nhận ra lệch (ITU-R BT.1359: tiếng sớm ~45 ms, trễ
~125 ms). Căn cứ: docs/v003_data.md §4.3.4.
Pseudo-fake thay đoạn tiếng (chỉ khi train, model bật `pseudo_fake`): với clip real, xác suất p
thay đặc trưng tiếng của một đoạn 0,3-3 s bằng đoạn cùng chỗ của một clip real khác cùng người
(như ô ② sham nhưng sinh vô hạn khi nạp); nhãn mismatch = 1 trên đoạn, AI / audio / visual = 0.
Căn cứ: Astrid et al. 2025 (thay đoạn: AUC chéo bộ 61 → 80 %), docs/v003_model.md §5.
Tham chiếu người (enrollment): R clip real của cùng speaker trong cùng split, luôn bỏ clip gốc
(source_clip_id) của chính sample. Khi train lấy ngẫu nhiên; khi đánh giá cố định: R clip real đầu
tiên theo sample_id trừ clip gốc của sample (`fixed_references`). Trước 25/10 bản đánh giá không
bỏ clip gốc: 15 % mẫu val / test được so với chính bản gốc (real và fake cùng nguồn) — rò.
Chế độ "ref" của báo cáo (`evaluate.ref_norm`) vẫn dùng R clip đầu tiên làm thống kê và không tính
điểm các nhóm clip đó.
Biến thể nén (v005, RESEARCH_WORKLOG/v005_model.md): `variants` = {tên: [thư mục cache]}
của cùng mẫu sau khi nén lại video + tiếng (vd "c480": 480p CRF 30, AAC 64k; prepare stage
`reencode`). Train: xác suất `variant_p` lấy CẢ mọi đặc trưng của mẫu từ bản nén (real và fake như
nhau); `pair=True` trả cả bản sạch lẫn bản nén của cùng mẫu với cùng ngẫu nhiên (cắt, trượt tiếng,
augment) để model học nhất quán. Đánh giá: `variant=<tên>` nạp toàn bộ từ bản đó.
"""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from vn_av_df.dataset import temporal_targets

KEYS = {
    "visual": "visual_fake_intervals",
    "audio": "audio_fake_intervals",
    "mismatch": "av_mismatch_intervals",
}
AUDIO_KEYS = ("a", "ssl", "wave", "prosody")
STATIC_KEYS = ("lipinc",)  # không theo frame: không cắt theo độ dài, không crop
PSEUDO_RANGE_S, PSEUDO_MARGIN_S = (0.3, 3.0), 0.5
# Lớp của đầu "loại thao tác" (M3): ô của mẫu; pseudo-fake (thay đoạn tiếng thật) xếp như sham.
CELL_INDEX = {c: i for i, c in enumerate(("real", "sham", "voice", "source", "donor", "both"))}


def frame_targets(row, frames, lag=0):
    """Nhãn theo frame; mismatch theo training/sync_labels.py (v004: có thể mask, phụ thuộc độ
    trượt tiếng `lag` đã áp khi nạp). `mismatch_valid` = frame được tính loss mismatch."""
    from vn_av_df.training import sync_labels

    times = np.arange(frames) / 25
    out = {name: temporal_targets(row, times, 0.04, key) for name, key in KEYS.items()}
    out["ai"] = temporal_targets(row, times, 0.04, "fake_intervals")
    out = {k: np.clip(v, 0, 1).astype(np.float32) for k, v in out.items()}
    out["mismatch"], out["mismatch_valid"] = sync_labels.frame_targets(row, frames, lag)
    return out


def video_targets(row, lag=0):
    """Nhãn video; mismatch = −1 khi không xác định (mask, bỏ khỏi loss video)."""
    from vn_av_df.training import sync_labels

    state = sync_labels.video_state(row, lag)
    return {
        "ai": float(row["label"]),
        "visual": float(bool(row.get("synthetic_visual"))),
        "audio": float(bool(row.get("synthetic_audio"))),
        "mismatch": -1.0 if state is None else float(state),
    }


def lag_audio(item, lag):
    """Tiếng trễ `lag` frame so với hình (lag < 0: sớm); đệm 0 ở mép và mask frame đó."""
    if not lag:
        return item
    keys = [k for k in AUDIO_KEYS if k in item]
    if not keys:
        return item
    n = len(item[keys[0]])
    for key in keys:
        x = item[key]
        pad = np.zeros((abs(lag), *x.shape[1:]), x.dtype)
        item[key] = np.concatenate([pad, x[:-lag]]) if lag > 0 else np.concatenate([x[-lag:], pad])
    keep = np.ones(n, bool)
    if lag > 0:
        keep[:lag] = False
    else:
        keep[lag:] = False
    item["lag_valid"] = keep
    return item


def replace_audio(item, other, targets, rng):
    """Thay đặc trưng tiếng của một đoạn 0,3-3 s bằng đoạn cùng chỗ của clip khác; trả đoạn."""
    keys = [k for k in AUDIO_KEYS if k in item and k in other]
    if not keys:
        return None
    n, m = len(item[keys[0]]), len(other[keys[0]])
    margin = int(round(PSEUDO_MARGIN_S * 25))
    low, high = (int(round(s * 25)) for s in PSEUDO_RANGE_S)
    high = min(high, n - 2 * margin, m)
    if high < low:
        return None
    size = rng.randint(low, high)
    start = rng.randint(margin, n - margin - size)
    source = min(start, m - size)
    for key in keys:
        item[key] = item[key].copy()
        item[key][start : start + size] = other[key][source : source + size]
    targets["mismatch"][start : start + size] = 1.0
    if "mismatch_valid" in targets:
        targets["mismatch_valid"][start : start + size] = True
    return start, start + size


def augment_mouth(mouth, rng, *others):
    """Xử lý lành tính áp như nhau cho real và fake (cùng tham số cho cả clip). `others`: các chồng ảnh
    khác của cùng khung (vùng tham chiếu `ref`) nhận đúng cùng tham số → trả tuple."""
    import cv2

    size = mouth.shape[1]
    quality = rng.randint(30, 95) if rng.random() < 0.7 else None
    factor = rng.uniform(0.5, 1.0) if rng.random() < 0.5 else None
    gamma = rng.uniform(0.7, 1.4) if rng.random() < 0.5 else None
    sharpen = rng.random() < 0.2
    blur = rng.uniform(0.3, 1.0) if not sharpen and rng.random() < 0.3 else None
    table = None if gamma is None else (((np.arange(256) / 255.0) ** gamma) * 255).astype(np.uint8)
    stacks = [s.copy() for s in (mouth, *others)]
    for out in stacks:
        for i in range(len(out)):
            x = out[i]
            if factor:
                small = max(8, int(size * factor))
                x = cv2.resize(
                    cv2.resize(x, (small, small), interpolation=cv2.INTER_AREA),
                    (size, size),
                    interpolation=cv2.INTER_LINEAR,
                )
            if blur:
                x = cv2.GaussianBlur(x, (0, 0), blur)
            if sharpen:
                x = cv2.addWeighted(x, 1.5, cv2.GaussianBlur(x, (0, 0), 1.0), -0.5, 0)
            if table is not None:
                x = cv2.LUT(x, table)
            if quality:
                ok, data = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, quality])
                x = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if ok else x
            out[i] = x
    return tuple(stacks) if others else stacks[0]


def fixed_references(pool, row, count):
    """Tham chiếu cố định khi đánh giá: `count` clip đầu của pool (đã xếp theo sample_id) trừ clip
    gốc của chính sample, để real và fake cùng nguồn không bao giờ được so với chính bản gốc."""
    return [r for r in pool if r["source_clip_id"] != row["source_clip_id"]][:count]


def enrollment(rows, count):
    """Mỗi (split, speaker): `count` clip real đầu tiên theo sample_id (None: tất cả, đã xếp)."""
    reals = defaultdict(list)
    for r in rows:
        if r.get("cell") == "real" and r.get("speaker_id"):
            reals[(r["split"], r["speaker_id"])].append(r)
    return {k: sorted(v, key=lambda r: r["sample_id"])[:count] for k, v in reals.items()}


# needs của model → encoder cần đọc (cùng bảng với prepare.NEEDS, không import để tránh vòng).
ENCODERS_FOR_NEED = {
    "av": ("avhubert",),
    "mouth": ("avhubert", "mouth"),
    "pose": ("avhubert", "mouth"),
    "lipinc": ("avhubert", "lipinc"),
    "ssl": ("xlsr",),
    "wave": ("wave",),
    # M3: ngữ điệu (từ tiếng 16 kHz) + đầu / lông mày (landmark của cache AV-HuBERT, tư thế đầu).
    "prosody": ("avhubert", "mouth", "wave"),
    # v005: vùng tham chiếu cùng khung (giữa hai mắt / gốc mũi, cùng cạnh hộp miệng) cho trace tương đối.
    "ref": ("avhubert", "mouth", "ref"),
}


def cache_roots(cfg):
    """cfg["cache"] là một thư mục hoặc danh sách thư mục (mỗi encoder có thể nằm ở dataset riêng)."""
    value = cfg["cache"]
    return [Path(p) for p in (value if isinstance(value, (list, tuple)) else [value])]


def cache_dir(cfg, encoder):
    """Thư mục cache của một encoder: thư mục gốc đầu tiên có nó (mặc định: gốc đầu tiên)."""
    roots = cache_roots(cfg)
    return next((r / encoder for r in roots if (r / encoder).is_dir()), roots[0] / encoder)


def cache_file(cfg, encoder, sample_id):
    """File cache của một mẫu: tìm ở mọi thư mục gốc (mỗi part một dataset cache riêng; cache 3 part
    ~60-75 GB không vừa một Output Kaggle 20 GB). Không thấy thì trả đường dẫn ở gốc đầu."""
    roots = cache_roots(cfg)
    for folder in roots:
        path = folder / encoder / f"{sample_id}.npz"
        if path.is_file():
            return path
    return roots[0] / encoder / f"{sample_id}.npz"


def cache_index(cfg, encoder, sample_ids):
    """{sample_id: file} cho cả tập, tìm một lần lúc dựng dataset (khỏi dò từng gốc mỗi lần nạp).
    Liệt kê mỗi thư mục một lần thay vì is_file từng mẫu × từng gốc (chậm trên /kaggle/input)."""
    roots = cache_roots(cfg)
    found = {}
    for root in roots:
        folder = root / encoder
        if not folder.is_dir():
            continue
        names = {p.name for p in folder.iterdir()}
        for sid in sample_ids:
            if sid not in found and f"{sid}.npz" in names:
                found[sid] = folder / f"{sid}.npz"
    return {sid: found.get(sid, roots[0] / encoder / f"{sid}.npz") for sid in sample_ids}


def speaker_index(rows):
    """Chỉ số người của split train (cho đầu phân loại người qua gradient reversal)."""
    names = sorted({r["speaker_id"] for r in rows if r["split"] == "train" and r.get("speaker_id")})
    return {name: i for i, name in enumerate(names)}


class BenchData(torch.utils.data.Dataset):
    """needs ⊆ {"av", "ssl", "mouth", "pose", "lipinc", "wave"}; references = số clip tham chiếu."""

    def __init__(
        self,
        cfg,
        rows,
        needs,
        augment=False,
        lag_augment=False,
        references=0,
        ref_frames=50,
        fixed_references=False,
        seed=0,
        crop_frames=None,
        pseudo_fake=0.0,
        rawboost=False,
        speakers=None,
        fixed_lag=None,
        variants=None,
        variant_p=0.0,
        pair=False,
        variant=None,
    ):
        self.rows, self.needs, self.augment = list(rows), set(needs), augment
        # Trượt tiếng cố định k frame cho mọi mẫu (phép thử trễ khi đánh giá, v004 §7); None = không.
        self.fixed_lag = fixed_lag
        ids = [r["sample_id"] for r in self.rows]
        # Encoder model cần (+ avhubert cho mặt nạ / hộp miệng): file từng mẫu, ở gốc cache nào cũng được.
        wanted = {"avhubert"} | {e for need in self.needs for e in ENCODERS_FOR_NEED.get(need, ())}
        self.files = {e: cache_index(cfg, e, ids) for e in sorted(wanted)}
        # Biến thể nén: chỉ mẫu có đủ mọi encoder cần ở bản đó mới được chọn.
        self.variant_files, self.variant_ids = {}, {}
        for name, roots in (variants or {}).items():
            files, have = {}, None
            for e in sorted(wanted):
                files[e] = {}
                for root in [Path(r) for r in (roots if isinstance(roots, (list, tuple)) else [roots])]:
                    folder = root / e
                    if folder.is_dir():
                        names = {p.name for p in folder.iterdir()}
                        for sid in ids:
                            if sid not in files[e] and f"{sid}.npz" in names:
                                files[e][sid] = folder / f"{sid}.npz"
                have = set(files[e]) if have is None else have & set(files[e])
            self.variant_files[name], self.variant_ids[name] = files, have or set()
        self.variant_p, self.pair, self.variant = float(variant_p), bool(pair), variant
        if variant is not None and variant not in self.variant_files:
            raise ValueError(f"không có biến thể {variant}: {sorted(self.variant_files)}")
        self.cache_cfg = cfg
        self.lag_augment = lag_augment
        self.references, self.ref_frames, self.seed = references, ref_frames, seed
        self.crop_frames = crop_frames  # train: cắt ngẫu nhiên một đoạn (bộ nhớ, cửa sổ 4 s)
        self.pseudo_fake, self.rawboost = pseudo_fake, rawboost
        self.speakers = speakers or {}
        self.fixed = enrollment(self.rows, None) if fixed_references else None
        pools = defaultdict(list)
        for r in self.rows:
            if r.get("cell") == "real" and r.get("speaker_id"):
                pools[(r["split"], r["speaker_id"])].append(r)
        self.pools = pools
        self.epoch = 0
        self.lag_range = tuple((cfg.get("training") or {}).get("audio_lag_aug") or (0, 0))

    def cache_file(self, encoder, sid, variant=None):
        """File cache của mẫu: tra bảng dựng sẵn; mẫu ngoài bảng (demo nạp video tạm) thì tìm trực tiếp."""
        if variant is not None:
            return self.variant_files[variant][encoder][sid]
        found = self.files.get(encoder, {}).get(sid)
        return found if found is not None else cache_file(self.cache_cfg, encoder, sid)

    def __len__(self):
        return len(self.rows)

    def load(self, row, rng=None, frames=None, needs=None, with_lag=False, variant=None):
        """(đặc trưng, độ dài); with_lag=True: thêm độ trượt tiếng k đã áp (frame).
        variant: tên biến thể nén (mọi đặc trưng lấy từ bản đó); None = bản gốc."""
        if variant is not None and row["sample_id"] not in self.variant_ids.get(variant, ()):
            variant = None
        needs = self.needs if needs is None else set(needs)
        sid = row["sample_id"]
        item, lengths, static = {}, [], {}
        if "av" in needs:
            with np.load(self.cache_file("avhubert", sid, variant), allow_pickle=False) as z:
                item["a"], item["v"] = z["audio"].astype(np.float32), z["visual"].astype(np.float32)
                item["av_valid"] = (z["audio_valid"] & z["visual_valid"]).copy()
            lengths.append(len(item["a"]))
        if "ssl" in needs:
            with np.load(self.cache_file("xlsr", sid, variant), allow_pickle=False) as z:
                ssl = z["ssl"].astype(np.float32)
            item["ssl"] = ssl[:, None] if ssl.ndim == 2 else ssl  # [T, nhóm, 1024]
            lengths.append(len(item["ssl"]))
        if needs & {"mouth", "pose"}:
            with np.load(self.cache_file("mouth", sid, variant), allow_pickle=False) as z:
                if "mouth" in needs:
                    item["mouth"] = z["mouth"].copy()
                item["lips"] = z["lips"].astype(np.float32)
                item["pose"] = (
                    z["pose"].astype(np.float32)
                    if "pose" in z.files
                    else np.zeros((len(z["lips"]), 3), np.float32)
                )
                item["mouth_valid"] = z["valid"].copy()
                # Độ phân giải thật của vùng miệng (px, trước khi đưa về 96): tín hiệu chất lượng.
                side = float(z["native_side"]) if "native_side" in z.files else 100.0
                item["quality"] = np.full(
                    (len(z["lips"]), 1), np.log(max(side, 1.0) / 100.0), np.float32
                )
            if "ref" in needs:
                with np.load(self.cache_file("ref", sid, variant), allow_pickle=False) as z:
                    item["ref"] = z["ref"].copy()
                    item["mouth_valid"] = item["mouth_valid"] & z["valid"][: len(item["mouth_valid"])]
            if "mouth" in item and self.augment and rng is not None:
                if "ref" in item:
                    item["mouth"], item["ref"] = augment_mouth(item["mouth"], rng, item["ref"])
                else:
                    item["mouth"] = augment_mouth(item["mouth"], rng)
            lengths.append(len(item["lips"]))
        if "wave" in needs:
            with np.load(self.cache_file("wave", sid, variant), allow_pickle=False) as z:
                item["wave"] = z["wave"].astype(np.float32) / 32768.0
            lengths.append(len(item["wave"]))
        if "prosody" in needs:
            # Tính trước khi trượt tiếng: "prosody" thuộc AUDIO_KEYS nên trượt / thay đoạn cùng tiếng.
            from vn_av_df.training.prosody import head_features, prosody_features

            wave = item.get("wave")
            if wave is None:
                with np.load(self.cache_file("wave", sid, variant), allow_pickle=False) as z:
                    wave = z["wave"].astype(np.float32) / 32768.0
            item["prosody"] = prosody_features(wave)
            with np.load(self.cache_file("avhubert", sid, variant), allow_pickle=False) as z:
                landmarks, seen = z["landmarks"], z["visual_valid"]
            pose = item.get("pose")
            if pose is None:
                with np.load(self.cache_file("mouth", sid, variant), allow_pickle=False) as z:
                    pose = z["pose"] if "pose" in z.files else np.zeros((len(z["lips"]), 3))
            item["head"] = head_features(landmarks, pose, seen)
            lengths += [len(item["prosody"]), len(item["head"])]
        if "lipinc" in needs:
            with np.load(self.cache_file("lipinc", sid, variant), allow_pickle=False) as z:
                static["lipinc"] = z["frames"].copy()
                item["lipinc_valid"] = z["valid"].copy()
            lengths.append(len(item["lipinc_valid"]))
        lag = 0
        if self.fixed_lag is not None:
            lag = int(self.fixed_lag)
        elif self.lag_augment and rng is not None and self.lag_range != (0, 0):
            lag = rng.randint(*self.lag_range)
        item = lag_audio(item, lag)
        length = min(lengths + ([frames] if frames else []))
        item = {k: v[:length] for k, v in item.items()}
        valid = np.ones(length, bool)
        for key in ("av_valid", "mouth_valid", "lag_valid"):
            if key in item:
                valid &= item[key]
        item["valid"] = valid
        item.update(static)
        return (item, length, lag) if with_lag else (item, length)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.variant is not None:
            return self.build(index, self.variant)
        have = [n for n, ids in self.variant_ids.items() if row["sample_id"] in ids]
        pick = random.Random(f"{self.seed}:{self.epoch}:{row['sample_id']}:v")
        if have and self.pair:
            item = self.build(index, None)
            item["_pair"] = self.build(index, pick.choice(have))
            return item
        if have and self.variant_p > 0 and pick.random() < self.variant_p:
            return self.build(index, pick.choice(have))
        return self.build(index, None)

    def build(self, index, variant):
        """Một mẫu từ bản gốc (variant=None) hoặc bản nén; cùng index + epoch → cùng ngẫu nhiên."""
        row = self.rows[index]
        rng = random.Random(f"{self.seed}:{self.epoch}:{row['sample_id']}")
        item, length, lag = self.load(row, rng, with_lag=True, variant=variant)
        targets = frame_targets(row, length, lag)
        video = video_targets(row, lag)
        pseudo = False
        if self.pseudo_fake and row.get("cell") == "real" and rng.random() < self.pseudo_fake:
            key = (row["split"], row.get("speaker_id"))
            pool = [
                r for r in self.pools.get(key, []) if r["source_clip_id"] != row["source_clip_id"]
            ]
            if pool:
                other, _ = self.load(
                    rng.choice(pool),
                    None,
                    needs=self.needs & {"av", "ssl", "wave", "prosody"},
                    variant=variant,
                )
                if replace_audio(item, other, targets, rng):
                    video["mismatch"] = 1.0
                    pseudo = True
        item.update({f"y_{k}": v for k, v in targets.items()})
        if self.crop_frames and length > self.crop_frames:
            start = rng.randrange(length - self.crop_frames + 1)
            item = {
                k: v if k in STATIC_KEYS else v[start : start + self.crop_frames]
                for k, v in item.items()
            }
        if self.rawboost and "wave" in item and rng is not None:
            from vn_av_df.training.rawboost import ssi_additive_noise

            flat = item["wave"].reshape(-1)
            noisy = ssi_additive_noise(flat, np.random.default_rng(rng.randrange(2**32)))
            item["wave"] = noisy.reshape(item["wave"].shape)
        item["index"] = index
        item["degraded"] = float(variant in self.variant_ids and row["sample_id"] in self.variant_ids[variant])
        item["video"] = video
        # Đầu offset (M3) học ĝ ≈ độ trượt k chỉ trên clip real nguyên vẹn (môi thật, tiếng thật).
        item["lag_shift"] = float(lag)
        item["offset_valid"] = row.get("cell") == "real" and not pseudo
        # Đầu loại thao tác (M3): ô của mẫu; pseudo-fake = tiếng thật của clip khác → như sham.
        item["cell_index"] = CELL_INDEX["sham"] if pseudo else CELL_INDEX.get(row.get("cell"), -1)
        item["speaker"] = (
            self.speakers.get(row.get("speaker_id"), -1) if row["split"] == "train" else -1
        )
        if self.references:
            key = (row["split"], row.get("speaker_id"))
            if self.fixed is not None:
                refs = fixed_references(self.fixed.get(key, []), row, self.references)
            else:
                pool = [
                    r
                    for r in self.pools.get(key, [])
                    if r["source_clip_id"] != row["source_clip_id"]
                ]
                refs = rng.sample(pool, min(self.references, len(pool)))
            item["refs"] = [self.load(r, None, self.ref_frames)[0] for r in refs[: self.references]]
        return item


FRAME_KEYS = ("a", "v", "ssl", "lips", "pose", "wave", "prosody", "head", "quality")


def collate(batch):
    """Đệm theo T dài nhất; mask = valid & trong độ dài thật. Mẫu có "_pair" (bản nén cùng mẫu):
    batch thành [B bản gốc, B bản nén], out["pair"] = B."""
    pair = len(batch) if "_pair" in batch[0] else 0
    if pair:
        batch = [{k: v for k, v in b.items() if k != "_pair"} for b in batch] + [b["_pair"] for b in batch]
    size = max(len(b["valid"]) for b in batch)

    def stack(key, dtype=torch.float32):
        arrays = [b[key] for b in batch]
        shape = (len(arrays), size, *arrays[0].shape[1:])
        out = np.zeros(shape, arrays[0].dtype)
        for i, x in enumerate(arrays):
            out[i, : len(x)] = x
        return torch.as_tensor(out, dtype=dtype)

    out = {
        "mask": stack("valid", torch.bool),
        "index": torch.tensor([b["index"] for b in batch]),
        "speaker": torch.tensor([b.get("speaker", -1) for b in batch]),
    }
    for key in FRAME_KEYS:
        if key in batch[0]:
            out[key] = stack(key)
    for key in ("mouth", "ref"):
        if key in batch[0]:
            out[key] = stack(key, torch.uint8)
    if "lipinc" in batch[0]:
        out["lipinc"] = torch.as_tensor(np.stack([b["lipinc"] for b in batch]))
    for key in ("y_ai", "y_visual", "y_audio", "y_mismatch"):
        out[key] = stack(key)
    if "y_mismatch_valid" in batch[0]:
        out["y_mismatch_valid"] = stack("y_mismatch_valid", torch.bool)
    out["lag_shift"] = torch.tensor([float(b.get("lag_shift", 0.0)) for b in batch])
    out["offset_valid"] = torch.tensor([bool(b.get("offset_valid", False)) for b in batch])
    out["cell_index"] = torch.tensor([int(b.get("cell_index", -1)) for b in batch])
    out["degraded"] = torch.tensor([float(b.get("degraded", 0.0)) for b in batch])
    if pair:
        out["pair"] = pair
    out["video"] = {k: torch.tensor([b["video"][k] for b in batch]) for k in batch[0]["video"]}
    if "refs" in batch[0]:
        count = max(len(b["refs"]) for b in batch)
        out["ref_count"] = torch.tensor([len(b["refs"]) for b in batch])
        if count:
            flat = [r for b in batch for r in b["refs"]] or []
            ref_size = max(len(r["valid"]) for r in flat)
            refs = {}
            for key in ("a", "v", "ssl", "mouth", "lips", "pose", "prosody", "head", "valid"):
                if key not in flat[0]:
                    continue
                sample = flat[0][key]
                dtype = sample.dtype
                tensor = np.zeros((len(batch), count, ref_size, *sample.shape[1:]), dtype)
                for i, b in enumerate(batch):
                    for j, r in enumerate(b["refs"]):
                        tensor[i, j, : len(r[key])] = r[key]
                refs[key] = torch.as_tensor(tensor)
            out["refs"] = refs
    return out


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device) if torch.is_tensor(batch) else batch
