"""Trích feature cho dataset v003 đã finalize: avhubert → mouth / lipinc (cần landmark) → xlsr, wave.

Chỉ trích encoder mà các model được chọn cần (`encoders_for`). cfg["cache"] có thể là danh sách:
ghi vào thư mục đầu tiên, đọc cache avhubert (cho mouth / lipinc) ở thư mục nào có nó, để mỗi
encoder trích ở một phiên Kaggle riêng rồi lưu thành dataset riêng (Output một phiên có giới hạn).

Cache: <cfg["cache"]>/<encoder>/<sample_id>.npz (+ .json). Chạy lại thì bỏ qua cache còn khớp
chữ ký; dừng giữa chừng không làm hỏng cache (ghi file tạm rồi đổi tên).
"""

import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from vn_av_df import tracking
from vn_av_df.common.runtime import environment, read_json, write_json
from vn_av_df.dataset import media_path, training_dataset
from vn_av_df.progress import Progress, duration, system_status
from vn_av_df.training.data import cache_dir, cache_roots

ORDER = ("avhubert", "mouth", "lipinc", "xlsr", "wave")
# needs của model → encoder phải có (mouth, lipinc đọc landmark / hộp miệng từ cache avhubert).
NEEDS = {
    "av": ("avhubert",),
    "mouth": ("avhubert", "mouth"),
    "pose": ("avhubert", "mouth"),
    "lipinc": ("avhubert", "lipinc"),
    "ssl": ("xlsr",),
    "wave": ("wave",),
}


def encoders_for(models):
    """Encoder cần cho danh sách model (theo thứ tự trích)."""
    from vn_av_df.training.models import build

    needed = set()
    for name in models:
        for need in build(name, dim=8, hidden=8, assets={"load_pretrained": False}).needs:
            needed.update(NEEDS[need])
    return [e for e in ORDER if e in needed]


def make(cfg, name, device):
    spec = {**cfg["encoders"][name], **({"device": device} if device else {})}
    if name == "avhubert":
        from vn_av_df.features.avhubert import AVHubertVideoEncoder

        return AVHubertVideoEncoder(spec)
    if name == "xlsr":
        from vn_av_df.features.encoders import XLSREncoder

        return XLSREncoder(spec)
    if name == "mouth":
        from vn_av_df.features.encoders import MouthEncoder

        return MouthEncoder(spec)
    if name == "lipinc":
        from vn_av_df.features.encoders import LipincEncoder

        return LipincEncoder(spec)
    if name == "wave":
        from vn_av_df.features.encoders import WaveEncoder

        return WaveEncoder(spec)
    raise ValueError(f"Unknown encoder {name}")


def prepare(cfg, encoders=ORDER, factory=None):
    rows, selection = training_dataset(cfg, verify_media=False)
    cache = cache_roots(cfg)[0]
    reports = {}
    chosen = [e for e in ORDER if e in encoders]
    run = tracking.open_run(
        cfg,
        "prepare",
        f"{cache.name}-prepare-{'+'.join(chosen)}",
        group=cache.name,
        config={
            "encoders": {e: cfg["encoders"][e] for e in chosen},
            "selection": selection,
            "samples": len(rows),
        },
        folder=cache,
    )
    print(
        f"prepare: {len(rows)} mẫu, encoder {chosen} → {cache} | "
        f"thiết bị {cfg.get('prepare_devices') or ['cpu']} | {system_status()}",
        flush=True,
    )
    for index, name in enumerate(chosen, 1):
        devices = list(cfg.get("prepare_devices") or [None])
        copies = int(cfg["encoders"][name].get("workers_per_device", 1))
        pool_devices = [d for d in devices for _ in range(copies)]
        folder = cache / name
        folder.mkdir(parents=True, exist_ok=True)
        cached = sum(1 for r in rows if (folder / f"{r['sample_id']}.json").is_file())
        print(
            f"== [{index}/{len(chosen)}] {name}: {len(rows)} mẫu, đã có {cached} file cache "
            f"(khớp chữ ký thì bỏ qua) | {len(pool_devices)} worker trên {pool_devices}",
            flush=True,
        )
        began = time.perf_counter()
        available = queue.Queue()
        with Progress(f"{name}: tạo worker", len(pool_devices), unit="worker", every_n=1) as made:
            for device in pool_devices:
                available.put((factory or make)(cfg, name, device))
                made.step()
        startup = time.perf_counter() - began
        groups = defaultdict(list)
        for row in rows:
            groups[row.get("source_clip_id") or row["sample_id"]].append(row)
        guard, done, first = threading.Lock(), [0], [None]

        def extract(group):
            encoder = available.get()
            try:
                for row in group:
                    request = {"video": str(media_path(cfg["dataset"], row))}
                    if name in ("mouth", "lipinc"):
                        boxes = cache_dir(cfg, "avhubert")
                        request["box_cache"] = str(boxes / f"{row['sample_id']}.npz")
                    try:
                        encoder.extract(request, folder / f"{row['sample_id']}.npz")
                    except Exception as exc:
                        print(
                            f"{name}: LỖI ở {row['sample_id']} ({request['video']}): {exc}",
                            flush=True,
                        )
                        raise
                    with guard:
                        done[0] += 1
                        if first[0] is None:
                            first[0] = time.perf_counter() - began
                        if done[0] % 25 == 0 or done[0] == len(rows):
                            spent = time.perf_counter() - began
                            run.log(
                                {
                                    f"{name}/done": done[0],
                                    f"{name}/samples_per_min": done[0] / max(spent, 1e-6) * 60,
                                }
                            )
                    progress.step()
            finally:
                available.put(encoder)

        try:
            with (
                Progress(name, len(rows)) as progress,
                ThreadPoolExecutor(max_workers=len(pool_devices)) as pool,
            ):
                for future in [pool.submit(extract, g) for g in groups.values()]:
                    future.result()
        finally:
            while not available.empty():
                getattr(available.get(), "close", lambda: None)()
        elapsed = time.perf_counter() - began
        quality = {**cache_quality(name, folder, rows), **log_cache(run, name, folder, rows)}
        write_json(
            folder / "extraction.json",
            {
                "samples": len(rows),
                "selection": selection,
                "elapsed_s": elapsed,
                "startup_s": startup,
                "first_sample_s": first[0],
                "devices": pool_devices,
                "environment": environment(),
                **quality,
            },
        )
        reports[name] = {"samples": len(rows), "elapsed_s": round(elapsed, 1), **quality}
        run.summary(
            {
                f"{name}/elapsed_h": elapsed / 3600,
                f"{name}/samples": len(rows),
                f"{name}/startup_s": startup,
                f"{name}/first_sample_s": first[0],
            }
        )
        print(
            f"== {name} xong: {len(rows)} mẫu trong {duration(elapsed)} (tạo worker "
            f"{duration(startup)}, mẫu đầu sau {duration(first[0] or 0)}) | chất lượng {quality}",
            flush=True,
        )
        run.artifact(f"{cache.name}-{name}-extraction", "extraction", [folder / "extraction.json"])
    run.finish()
    return reports


def cache_quality(name, folder, rows):
    """Dữ liệu bị loại / xử lý khác luật gốc (Pineau: giải thích dữ liệu bị loại), ghi vào
    extraction.json: LIPINC-V2 gốc bỏ clip không tìm đủ frame theo luật, ở đây chọn theo luật
    dự phòng (`fallback`) và đếm lại."""
    if name != "lipinc":
        return {}
    flags, by_cell = [], defaultdict(list)
    for r in rows:
        meta = folder / f"{r['sample_id']}.json"
        if meta.is_file():
            flag = bool(read_json(meta).get("fallback", False))
            flags.append(flag)
            by_cell[r.get("cell")].append(flag)
    return {
        "lipinc_fallback": {
            "samples": len(flags),
            "fallback": int(sum(flags)),
            "ratio": float(np.mean(flags)) if flags else None,
            "ratio_by_cell": {c: float(np.mean(v)) for c, v in sorted(by_cell.items())},
        }
    }


def log_cache(run, name, folder, rows):
    """Chất lượng cache: tỉ lệ frame hợp lệ; với AV-HuBERT thêm kiểm chéo độ lệch tiếng-hình
    trên clip real (đã sửa về 0 ở data pipeline bằng SyncNet; ở đây đo lại bằng đặc trưng
    AV-HuBERT theo cách của SpeechForensics). Trả số liệu tóm tắt để ghi vào extraction.json."""
    if name not in ("avhubert", "mouth"):
        return {}
    valid, lags, peaks = [], [], []
    for row in rows:
        path = folder / f"{row['sample_id']}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            if name == "avhubert":
                ok = z["audio_valid"] & z["visual_valid"]
                valid.append(float(ok.mean()))
                if row.get("cell") == "real":
                    lag, peak = tracking.lag_profile(z["audio"], z["visual"], ok)
                    if lag is not None:
                        lags.append(lag)
                        peaks.append(peak)
            elif name == "mouth":
                valid.append(float(z["valid"].mean()))
    out = {}
    run.hist(f"{name}/valid_frame_ratio", valid)
    if valid:
        out["mean_valid_ratio"] = float(np.mean(valid))
        out["clips_under_half_valid"] = int(np.sum(np.asarray(valid) < 0.5))
        run.summary({f"{name}/mean_valid_ratio": out["mean_valid_ratio"]})
    if lags:
        lags = np.asarray(lags)
        run.hist("sync_check/avhubert_lag_real_frames", lags)
        run.hist("sync_check/avhubert_peak_real", peaks)
        out["sync_check"] = {
            "real_clips": len(lags),
            "share_within_1_frame": float(np.mean(np.abs(lags) <= 1)),
            "median_lag": float(np.median(lags)),
        }
        run.summary({f"sync_check/{k}": v for k, v in out["sync_check"].items()})
    return out
