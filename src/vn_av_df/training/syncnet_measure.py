"""Đo đường cong khớp SyncNet cho nhãn mismatch v004 (chạy local, CPU) → `<part>/sync_syncnet.json`.

Đo mọi clip real và mọi file lip-sync bản **full** của các part đang chọn (bản partial dùng lại số
đo của bản full cùng lần chạy generator, training/sync_labels.py). SyncNet và YuNet giống hệt
bước cắt của data pipeline (`vn_av_data.features.syncnet`, cấu hình `data_pipeline/configs`).

Mỗi mẫu lưu cả đường cong khoảng cách 31 giá trị (chỉ số i ↔ tiếng trễ i − 15 frame), không chỉ
độ lệch tốt nhất: luật nhãn cần khoảng cách **tại lag 0** và tại các độ trượt khi augment.
Trước khi đo, tự kiểm quy ước dấu trên một clip real: trượt tiếng biết trước +3 frame thì đỉnh
phải dịch +3 (±1), như phép kiểm 180/180 ở docs/v003_data.md §4.3.2. Chạy lại thì đo tiếp phần
chưa có; ghi file sau mỗi 25 mẫu.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vn_av_df.common.runtime import read_json, write_json
from vn_av_df.training.sync_labels import FILES, LIPSYNC_CELLS, MAX_LAG

SELF_CHECK_SHIFT = 3
_METER = None


def data_root():
    return Path(__file__).resolve().parents[3] / "data_pipeline"


def make_meter(threads=4):
    from vn_av_data.common.runtime import load_config
    from vn_av_data.features.syncnet import SyncMeter

    cut = load_config(data_root() / "configs/data.yaml")["cut"]
    return SyncMeter(data_root() / cut["syncnet_model"], data_root() / cut["face_model"], threads)


def _init(threads):
    global _METER
    _METER = make_meter(threads)


def decode_clip(path):
    from vn_av_data.data.media import decode

    return decode(path, max_duration=10, max_side=640, sample_rate=16000)


def curve_of(meter, frames, pcm):
    curve = meter.curve(frames, pcm)
    return None if curve is None else [round(float(x), 5) for x in curve]


def _measure(path):
    media = decode_clip(path)
    return curve_of(_METER, media["frames"], media["pcm"]), int(len(media["frames"]))


def self_check(meter, path, shift=SELF_CHECK_SHIFT):
    """Trượt tiếng +shift frame (đệm lặng ở đầu = tiếng trễ) → đỉnh SyncNet phải dịch +shift."""
    from vn_av_data.features.syncnet import SAMPLES_PER_FRAME as step

    media = decode_clip(path)
    frames, pcm = media["frames"], np.asarray(media["pcm"])
    delayed = np.concatenate([np.zeros(shift * step, pcm.dtype), pcm])[: len(pcm)]
    before, after = meter.curve(frames, pcm), meter.curve(frames, delayed)
    if before is None or after is None:
        return None
    lag0, lag1 = int(np.argmin(before)) - MAX_LAG, int(np.argmin(after)) - MAX_LAG
    result = {"clip": Path(path).name, "shift": shift, "lag_before": lag0, "lag_after": lag1}
    if abs((lag1 - lag0) - shift) > 1:
        raise RuntimeError(f"SyncNet tự kiểm quy ước dấu thất bại: {result}")
    return result


def wanted(row):
    return row.get("cell") == "real" or (
        row.get("cell") in LIPSYNC_CELLS and row.get("kind") == "full"
    )


def measure_part(folder, workers=1, threads=4, limit=None):
    from vn_av_df.dataset import load_dataset

    folder = Path(folder)
    out = folder / FILES["syncnet"]
    data = read_json(out) if out.is_file() else {}
    samples = dict(data.get("samples") or {})
    rows = [r for r in load_dataset(folder, verify_media=False) if wanted(r)]
    todo = [r for r in rows if r["sample_id"] not in samples][: limit or None]
    print(
        f"syncnet {folder.name}: {len(rows)} mẫu cần đo (real + lip-sync full), đã có "
        f"{len(rows) - len([r for r in rows if r['sample_id'] not in samples])}, đo {len(todo)}",
        flush=True,
    )
    check = data.get("self_check")
    if check is None:
        meter = make_meter(threads)
        for r in rows:
            if r.get("cell") == "real":
                check = self_check(meter, folder / r["video"])
                if check:
                    break
        print(f"syncnet tự kiểm quy ước dấu: {check}", flush=True)

    def save():
        write_json(
            out,
            {
                "meter": "syncnet_v2",
                "convention": "index i <-> audio lag i - 15",
                "max_lag": MAX_LAG,
                "self_check": check,
                "samples": samples,
            },
        )

    began, done = time.monotonic(), 0
    paths = [str(folder / r["video"]) for r in todo]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(threads,)) as pool:
        for row, (curve, frames) in zip(todo, pool.map(_measure, paths)):
            samples[row["sample_id"]] = {"dist": curve, "frames": frames}
            done += 1
            if done % 25 == 0 or done == len(todo):
                save()
                rate = done / max(1e-6, time.monotonic() - began)
                print(
                    f"syncnet {folder.name}: {done}/{len(todo)} | {rate * 60:.1f} mẫu/phút | "
                    f"còn ~{(len(todo) - done) / max(rate, 1e-6) / 60:.0f} phút",
                    flush=True,
                )
    save()
    failed = sum(1 for v in samples.values() if v.get("dist") is None)
    return {"part": folder.name, "samples": len(samples), "no_face": failed, "file": str(out)}


def measure(cfg, workers=1, threads=4, limit=None):
    from vn_av_df.dataset import selected_parts

    return [measure_part(folder, workers, threads, limit) for _, folder in selected_parts(cfg)]
