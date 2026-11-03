"""Nhãn mismatch v004: lệch nội dung theo cách tạo + lệch đo được ở ô lip-sync (0 / 1 / mask).

Căn cứ và luật: RESEARCH_WORKLOG/v004_model.md §5 (user chốt 31/10 - 01/11/2026).

- Lệch **nội dung** (② sham, ③ voice, pseudo-fake): 1 trên đoạn `av_mismatch_intervals`, như v003.
- Đoạn **môi do generator vẽ** (④⑤⑥): đo bằng đường cong khớp theo độ trễ của bản full cùng lần
  chạy generator, ở hai thước độc lập: AV-HuBERT (cos, `avhubert/sync_profile.json`, tính lúc
  prepare) và SyncNet (khoảng cách, `<part>/sync_syncnet.json`, đo local). "Độ tệ" tại lag 0 so
  với phân bố real của split train: > phân vị 99 % ở **mọi** thước → 1; ≤ phân vị 95 % ở mọi
  thước → 0; còn lại → mask. Chỉ tin khi clip real gốc đi cặp đo được (độ nhọn ≥ phân vị 10 % của
  real): không đo được thì mask (Synchformer: có cặp không đồng bộ hoá được).
- Phần **môi thật + tiếng thật**: độ trượt |k| ≤ `near_shift` (sai số đo, dưới ngưỡng cảm nhận) → 0;
  trượt xa hơn thì dùng đường cong clip real gốc như trên.
- Trượt tiếng k frame khi train (dương = tiếng trễ, `data.lag_audio`): độ tệ tại lag 0 sau trượt
  = độ tệ của đường cong gốc tại lag −k.
- Vì sao đo tại lag 0, không theo đỉnh: LSE-C / LSE-D của SyncNet lấy ở độ lệch tốt nhất nên bỏ
  qua trễ cố định (syncnet_python); Wav2Lip được SyncNet chấm tốt hơn cả real (Wav2Lip Table 1),
  nên cần cả hai thước đồng ý.

Chế độ (`cfg["mismatch_labels"]["mode"]`): "construction" (v003: ô lip-sync = 0, không chú thích),
"mask_lipsync" (dự phòng không cần đo: ô lip-sync → mask), "measured" (luật trên).
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MAX_LAG = 15
METERS = ("avhubert", "syncnet")
RULE_VERSION = "mismatch-measured-v1"
LIPSYNC_CELLS = ("source", "donor", "both")
FILES = {"avhubert": "sync_profile.json", "syncnet": "sync_syncnet.json"}
CURVE_KEY = {"avhubert": "cos", "syncnet": "dist"}
DEFAULTS = {
    "mode": "construction",
    "meters": list(METERS),
    "near_shift": 1,
    "negative_quantile": 0.95,
    "positive_quantile": 0.99,
    "sharpness_quantile": 0.10,
    "min_reals": 30,
}


def spec_of(cfg):
    return {**DEFAULTS, **(cfg.get("mismatch_labels") or {})}


# ---------------------------------------------------------------- đường cong → độ tệ
def badness(meter, curve):
    """Độ tệ theo lag (cao = kém khớp): AV-HuBERT −cos, SyncNet khoảng cách."""
    curve = np.asarray(curve, float)
    return -curve if meter == "avhubert" else curve


def sharpness(meter, curve):
    """Độ nhọn của đỉnh khớp: AV-HuBERT max − median của cos; SyncNet median − min của khoảng
    cách (= Confidence của syncnet_python)."""
    curve = np.asarray(curve, float)
    if curve.size == 0 or np.all(np.isnan(curve)):
        return math.nan
    if meter == "avhubert":
        return float(np.nanmax(curve) - np.nanmedian(curve))
    return float(np.nanmedian(curve) - np.nanmin(curve))


def at_shift(bad, k):
    """Độ tệ tại lag 0 sau khi trượt tiếng k frame (= đường cong gốc tại lag −k)."""
    i = MAX_LAG - int(k)
    return float(bad[i]) if 0 <= i < len(bad) else math.nan


# ---------------------------------------------------------------- đọc số đo
def measurement_files(cfg):
    """(thước, file) có trên đĩa: AV-HuBERT ở mọi thư mục cache, SyncNet ở mọi part đang chọn."""
    from vn_av_df.dataset import selected_parts
    from vn_av_df.training.data import cache_roots

    found = [
        ("avhubert", root / "avhubert" / FILES["avhubert"])
        for root in cache_roots(cfg)
        if (root / "avhubert" / FILES["avhubert"]).is_file()
    ]
    for _, folder in selected_parts(cfg):
        if (Path(folder) / FILES["syncnet"]).is_file():
            found.append(("syncnet", Path(folder) / FILES["syncnet"]))
    for meter, path in (spec_of(cfg).get("files") or {}).items():  # chỉ định thêm (test / tay)
        for item in [path] if isinstance(path, (str, Path)) else path:
            found.append((meter, Path(item)))
    return found


def load_curves(cfg):
    """{thước: {sample_id: đường cong 31 giá trị}}."""
    from vn_av_df.common.runtime import read_json

    curves = {m: {} for m in METERS}
    for meter, path in measurement_files(cfg):
        data = read_json(path)
        if data.get("convention") not in (None, "index i <-> audio lag i - 15"):
            raise ValueError(f"{path}: quy ước đường cong khác ({data.get('convention')})")
        for sid, entry in (data.get("samples") or {}).items():
            curve = entry.get(CURVE_KEY[meter]) if isinstance(entry, dict) else entry
            if curve is not None and len(curve) == 2 * MAX_LAG + 1:
                curves[meter][sid] = [math.nan if x is None else float(x) for x in curve]
    return curves


def thresholds(rows, curves, meters, spec):
    """Ngưỡng từ clip real của split train (không dùng validation / test)."""
    reals = [r for r in rows if r.get("split") == "train" and r.get("cell") == "real"]
    out = {}
    for meter in meters:
        bad0, sharp = [], []
        for r in reals:
            curve = curves[meter].get(r["sample_id"])
            if curve is None:
                continue
            value = badness(meter, curve)[MAX_LAG]
            if math.isfinite(value):
                bad0.append(value)
            s = sharpness(meter, curve)
            if math.isfinite(s):
                sharp.append(s)
        if len(bad0) < spec["min_reals"]:
            raise ValueError(
                f"mismatch measured: thước {meter} chỉ có {len(bad0)} clip real train có số đo "
                f"(cần ≥ {spec['min_reals']}); chạy `sync-profile` / `syncnet` trước, hoặc bỏ "
                f"thước này khỏi mismatch_labels.meters"
            )
        out[meter] = {
            "negative": float(np.quantile(bad0, spec["negative_quantile"])),
            "positive": float(np.quantile(bad0, spec["positive_quantile"])),
            "sharpness": float(np.quantile(sharp, spec["sharpness_quantile"])),
            "reals": len(bad0),
        }
    return out


# ---------------------------------------------------------------- trạng thái
def region(row, curves, meters, thr):
    """Số đo của một mẫu: độ tệ theo lag ở mọi thước + có đo được không; None nếu thiếu thước."""
    if row is None:
        return None
    bad, measurable = {}, True
    for meter in meters:
        curve = curves[meter].get(row["sample_id"])
        if curve is None:
            return None
        bad[meter] = badness(meter, curve).tolist()
        s = sharpness(meter, curve)
        measurable &= math.isfinite(s) and s >= thr[meter]["sharpness"]
    return {"bad": bad, "measurable": bool(measurable), "sample_id": row["sample_id"]}


def measured_state(region_info, k, info, require_measurable=True):
    """1 / 0 / None (mask) cho một vùng tại độ trượt k. Vùng môi giả không cần đỉnh nhọn: đường
    cong phẳng mà độ tệ tại lag 0 cao vẫn là lệch (chỉ cần clip real gốc đo được)."""
    if region_info is None or (require_measurable and not region_info["measurable"]):
        return None
    values = {m: at_shift(region_info["bad"][m], k) for m in info["meters"]}
    if any(not math.isfinite(v) for v in values.values()):
        return None
    thr = info["thr"]
    if all(values[m] > thr[m]["positive"] for m in values):
        return 1
    if all(values[m] <= thr[m]["negative"] for m in values):
        return 0
    return None


def real_state(info, k):
    """Phần môi thật + tiếng thật, trượt k frame."""
    if abs(int(k)) <= info["near"]:
        return 0
    if info["mode"] != "measured":
        return None
    return measured_state(info.get("src"), k, info)


def generated_state(info, k):
    """Đoạn môi do generator vẽ, trượt k frame (đo trên bản full, gác bởi clip real gốc)."""
    if info["mode"] != "measured":
        return None
    src = info.get("src")
    if src is None or not src["measurable"]:
        return None
    return measured_state(info.get("gen"), k, info, require_measurable=False)


def covers_whole(intervals, duration):
    return bool(intervals) and sum(b - a for a, b in intervals) >= duration - 0.04


def video_state(row, k=0):
    """Nhãn mismatch cả video tại độ trượt k: 1 / 0 / None (mask)."""
    info = row.get("_mismatch")
    if info is None:
        return int(bool(row.get("av_mismatch_intervals")))
    if row.get("av_mismatch_intervals"):
        return 1
    states = []
    gen_intervals = info.get("gen_intervals") or []
    if gen_intervals:
        states.append(generated_state(info, k))
    if not covers_whole(gen_intervals, row["duration_s"]):
        states.append(real_state(info, k))
    if 1 in states:
        return 1
    return None if None in states else 0


def frame_targets(row, frames, k=0, step=0.04):
    """(nhãn [frames] trong [0, 1], hợp lệ [frames] bool) cho đầu mismatch."""
    from vn_av_df.dataset import temporal_targets

    times = np.arange(frames) / 25
    content = np.clip(temporal_targets(row, times, step, "av_mismatch_intervals"), 0, 1)
    info = row.get("_mismatch")
    if info is None:
        return content.astype(np.float32), np.ones(frames, bool)
    gen = np.zeros(frames, bool)
    if info.get("gen_intervals"):
        cover = temporal_targets(
            {"duration_s": row["duration_s"], "x": info["gen_intervals"]}, times, step, "x"
        )
        gen = cover > 0.5
    y = np.zeros(frames, np.float32)
    valid = np.ones(frames, bool)
    s_real = real_state(info, k)
    s_gen = generated_state(info, k) if gen.any() else None
    y[~gen], valid[~gen] = (s_real or 0), s_real is not None
    y[gen], valid[gen] = (s_gen or 0), s_gen is not None
    has = content > 0
    y[has], valid[has] = np.maximum(y[has], content[has]), True
    return y, valid


# ---------------------------------------------------------------- chú thích rows
def full_sibling(row, by_id):
    """Bản full cùng lần chạy generator (partial lấy từ cùng lần chạy, v003_generation §3.2)."""
    sid = row["sample_id"]
    if row.get("kind") == "partial" and sid.endswith("_partial"):
        return by_id.get(sid[: -len("_partial")] + "_full")
    return row


def annotate(cfg, rows):
    """Gắn `_mismatch` (thông tin luật) và `_mismatch_eval` (nhãn video tại k = 0) vào từng row.
    Chế độ "construction": không đổi gì (giống v003)."""
    spec = spec_of(cfg)
    mode = spec["mode"]
    if mode == "construction":
        return rows
    if mode not in ("mask_lipsync", "measured"):
        raise ValueError(f"mismatch_labels.mode không hợp lệ: {mode}")
    meters = [m for m in spec["meters"] if m in METERS]
    curves, thr = {m: {} for m in METERS}, {}
    if mode == "measured":
        if not meters:
            raise ValueError("mismatch measured cần ít nhất một thước trong meters")
        curves = load_curves(cfg)
        thr = thresholds(rows, curves, meters, spec)
    by_id = {r["sample_id"]: r for r in rows}
    real_of = {
        (r.get("split"), r.get("source_clip_id")): r for r in rows if r.get("cell") == "real"
    }
    out = []
    for row in rows:
        info = {"mode": mode, "near": int(spec["near_shift"]), "meters": meters, "thr": thr}
        if mode == "measured":
            src = real_of.get((row.get("split"), row.get("source_clip_id")))
            info["src"] = region(src, curves, meters, thr)
        if row.get("cell") in LIPSYNC_CELLS:
            info["gen_intervals"] = row.get("visual_fake_intervals") or []
            if mode == "measured":
                info["gen"] = region(full_sibling(row, by_id), curves, meters, thr)
        row = {**row, "_mismatch": info}
        row["_mismatch_eval"] = video_state(row, 0)
        out.append(row)
    return out


def summary(rows):
    """Bảng đếm nhãn video (k = 0) theo ô / generator, ngưỡng, và kiểm lối tắt độ trễ: AUC của độ
    tệ AV-HuBERT tại lag 0 (bản full lip-sync so với real cùng split); ngoài [0,3; 0,7] là cờ
    (luật audit của v003_generation §4.4)."""
    from sklearn.metrics import roc_auc_score

    counts = defaultdict(Counter)
    for r in rows:
        if "_mismatch" not in r:
            continue
        key = r["cell"] if r["cell"] not in LIPSYNC_CELLS else f"{r['cell']}/{r.get('generator')}"
        state = r["_mismatch_eval"]
        counts[f"{r['split']}/{key}"]["mask" if state is None else str(state)] += 1
    out = {
        "rule": RULE_VERSION,
        "counts": {k: dict(v) for k, v in sorted(counts.items())},
    }
    info = next((r["_mismatch"] for r in rows if "_mismatch" in r), None)
    if info is None or info["mode"] != "measured":
        return out
    out["thresholds"] = info["thr"]
    # Clip real train bị trượt k frame (dải augment) mang nhãn gì: cho thấy augment dạy gì.
    shifts = {}
    reals = [r for r in rows if r.get("split") == "train" and r.get("cell") == "real"]
    for k in range(-5, 8):
        tally = Counter(video_state(r, k) for r in reals)
        shifts[str(k)] = {"1": tally[1], "0": tally[0], "mask": tally[None]}
    out["real_train_by_shift"] = shifts
    audit = {}
    for split in ("train", "validation", "test"):
        reals = [
            r["_mismatch"]["src"]["bad"]["avhubert"][MAX_LAG]
            for r in rows
            if r.get("split") == split
            and r.get("cell") == "real"
            and r["_mismatch"].get("src")
            and "avhubert" in r["_mismatch"]["src"]["bad"]
        ]
        by_gen = defaultdict(list)
        for r in rows:
            gen = r["_mismatch"].get("gen")
            if (
                r.get("split") == split
                and r.get("kind") == "full"
                and gen
                and "avhubert" in gen["bad"]
            ):
                by_gen[r.get("generator")].append(gen["bad"]["avhubert"][MAX_LAG])
        for name, values in sorted(by_gen.items()):
            values = [v for v in values if math.isfinite(v)]
            finite = [v for v in reals if math.isfinite(v)]
            if values and finite:
                value = float(roc_auc_score([1] * len(values) + [0] * len(finite), values + finite))
                audit[f"{split}/{name}"] = {
                    "n": len(values),
                    "auc_vs_real": value,
                    "flag": not 0.3 <= value <= 0.7,
                }
    out["lag0_audit_avhubert"] = audit
    return out


# ---------------------------------------------------------------- số đo AV-HuBERT từ cache
def avhubert_profiles(cfg, rows, folder=None):
    """Đường cong cos theo lag cho mọi mẫu có cache AV-HuBERT → `<cache>/avhubert/sync_profile.json`.
    Gọi trong prepare (sau stage avhubert) hoặc riêng (`sync-profile`) với cache đã có."""
    from vn_av_df import tracking
    from vn_av_df.common.runtime import write_json
    from vn_av_df.training.data import cache_dir, cache_file

    given = folder is not None
    folder = Path(folder) if given else cache_dir(cfg, "avhubert")
    samples = {}
    for row in rows:
        sid = row["sample_id"]
        path = folder / f"{sid}.npz" if given else cache_file(cfg, "avhubert", sid)
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            ok = z["audio_valid"] & z["visual_valid"]
            curve = tracking.lag_curve(
                z["audio"].astype(np.float32), z["visual"].astype(np.float32), ok
            )
        samples[row["sample_id"]] = {
            "cos": [None if not math.isfinite(x) else round(float(x), 6) for x in curve],
            "frames": int(ok.sum()),
        }
    write_json(
        folder / FILES["avhubert"],
        {
            "meter": "avhubert",
            "convention": "index i <-> audio lag i - 15",
            "max_lag": MAX_LAG,
            "samples": samples,
        },
    )
    return {"samples": len(samples), "file": str(folder / FILES["avhubert"])}


def report(cfg):
    """Bước "labels" trước train (CPU): bảng đếm nhãn mismatch + kiểm lối tắt độ trễ + kiểm ngữ
    điệu–đầu của M3 → runs/<RUN>/sync_labels.json. Cần sync_syncnet.json trong mỗi part và
    avhubert/sync_profile.json trong cache (prepare)."""
    from vn_av_df import tracking
    from vn_av_df.common.runtime import write_json
    from vn_av_df.dataset import training_dataset
    from vn_av_df.training.prosody import audit

    rows, _ = training_dataset(cfg)
    result = summary(rows)
    result["meters_found"] = [[m, str(p)] for m, p in measurement_files(cfg)]
    runs = Path(cfg["runs"])
    # Phần nhãn chỉ mất vài chục giây: ghi và in ngay, phép kiểm dưới đây chậm hơn nhiều.
    write_json(runs / "sync_labels.json", result)
    print(json.dumps(result, indent=1, ensure_ascii=False)[:20000], flush=True)
    # M3: ngữ điệu ↔ đầu có tách real với ô thay tiếng không (không học tham số, trước train).
    result["prosody_head_audit"] = audit(cfg, rows)
    write_json(runs / "sync_labels.json", result)
    run = tracking.open_run(
        cfg, "labels", f"{runs.name}-sync-labels", group=runs.name, folder=runs, with_setup=True
    )
    run.artifact(f"labels-{runs.name}", "labels", [runs / "sync_labels.json"])
    run.finish()
    return result
