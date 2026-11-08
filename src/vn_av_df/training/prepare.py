"""Trích feature cho dataset v003 đã finalize: landmarks → avhubert, mouth, lipinc → xlsr, wave.

Chỉ trích encoder mà các model được chọn cần (`encoders_for`). cfg["cache"] có thể là danh sách:
ghi vào thư mục đầu tiên, đọc cache khác (landmark cho avhubert / mouth / lipinc) ở thư mục nào có
nó, để mỗi nhóm encoder trích ở một phiên Kaggle riêng rồi lưu thành dataset riêng.

Encoder chỉ cần CPU (`cpu_only`: landmarks, mouth, lipinc, wave) chạy số worker = số nhân CPU, trên
phiên Kaggle không GPU (không tốn quota GPU). dlib dò mặt chiếm ~86 % thời gian của avhubert cũ
(đo 24/10: 7,1 / 8,3 s mỗi mẫu), nên tách thành stage `landmarks` riêng.

Cache: <cfg["cache"]>/<encoder>/<sample_id>.npz (+ .json). Chạy lại thì bỏ qua cache còn khớp
chữ ký; dừng giữa chừng không làm hỏng cache (ghi file tạm rồi đổi tên).

Biến thể nén (v005, RESEARCH_WORKLOG/v005_model.md): cfg["variant"] = {"name": "c480",
"height": 480, "crf": 30, "audio_bitrate": "64k"} → trước các encoder, nén lại video + tiếng từng mẫu
(ffmpeg libx264 + AAC, giữ số frame) vào <cfg["cache"][0]>/clips/, rồi trích avhubert / mouth / xlsr
trên bản nén vào cfg["cache"][0]. Landmark (stage landmarks) dùng lại của bản gốc (các thư mục cache
sau trong danh sách): cùng tỉ lệ khung, toạ độ chuẩn hoá theo cạnh dài 640, cùng số frame.
"""

import hashlib
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from vn_av_df import tracking
from vn_av_df.common.runtime import environment, read_json, write_json
from vn_av_df.dataset import media_path, training_dataset
from vn_av_df.progress import Progress, duration, system_status
from vn_av_df.training.data import cache_file, cache_roots

ORDER = ("landmarks", "avhubert", "mouth", "lipinc", "xlsr", "wave")
# needs của model → encoder phải có (avhubert, mouth, lipinc đọc landmark / hộp miệng từ landmarks).
NEEDS = {
    "av": ("landmarks", "avhubert"),
    "mouth": ("landmarks", "avhubert", "mouth"),
    "pose": ("landmarks", "avhubert", "mouth"),
    "lipinc": ("landmarks", "avhubert", "lipinc"),
    "ssl": ("xlsr",),
    "wave": ("wave",),
    "prosody": ("landmarks", "avhubert", "mouth", "wave"),  # M3: ngữ điệu ↔ đầu (prosody.py)
}


def encoders_for(models):
    """Encoder cần cho danh sách model (theo thứ tự trích)."""
    from vn_av_df.training.models import build

    needed = set()
    for name in models:
        for need in build(name, dim=8, hidden=8, assets={"load_pretrained": False}).needs:
            needed.update(NEEDS[need])
    return [e for e in ORDER if e in needed]


def landmark_file(cfg, sample_id):
    """Landmark / hộp miệng của mẫu: cache stage landmarks; chưa có thì cache avhubert (cách cũ)."""
    path = cache_file(cfg, "landmarks", sample_id)
    return path if path.is_file() else cache_file(cfg, "avhubert", sample_id)


def make(cfg, name, device):
    spec = {**cfg["encoders"][name], **({"device": device} if device else {})}
    if name == "landmarks":
        from vn_av_df.features.avhubert import AVHubertVideoEncoder

        # Cùng worker / env / landmark model với avhubert, chỉ stage khác.
        return AVHubertVideoEncoder({**cfg["encoders"]["avhubert"], **spec}, stage="landmarks")
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


def ffmpeg_binary():
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def variant_selection(spec, rows):
    """Mẫu của biến thể: `splits` (mức chỉ để test: ["test"]) và `cells` (PR: real / sham / voice)."""
    splits, cells = spec.get("splits"), spec.get("cells")
    return [r for r in rows if (not splits or r["split"] in splits) and (not cells or r.get("cell") in cells)]


def processed_real_chain(spec, row):
    """Chuỗi xử lý PR của clip nguồn (cố định theo sha1 tên biến thể + source_clip_id; real / sham /
    voice cùng clip nguồn chung một chuỗi): (họ, u ∈ [0, 1] chọn tham số). Train / validation dùng
    `families`; test chia đôi theo hash: một nửa họ đã thấy, một nửa `heldout_families` (chưa thấy)."""
    digest = hashlib.sha1(f"{spec['name']}:{row['source_clip_id']}".encode()).digest()
    pool = spec["families"]
    if row["split"] == "test" and spec.get("heldout_families") and digest[2] % 2:
        pool = spec["heldout_families"]
    return pool[digest[0] % len(pool)], digest[1] / 255


def processed_real_middle(family, u, width, height):
    """Bước giữa của chuỗi PR, giả lập codec / xử lý trong đường ống generator mà không đổi nội
    dung miệng: (đuôi file, tham số ffmpeg). Đo 07/11 (RESEARCH_WORKLOG/v005_model.md §8)."""
    if family == "mp4v":
        return ".avi", ["-c:v", "mpeg4", "-q:v", str(3 + round(u * 5))]
    if family == "mjpeg":
        return ".avi", ["-c:v", "mjpeg", "-q:v", str(3 + round(u * 4)), "-pix_fmt", "yuvj420p"]
    if family == "rescale":
        f = 0.5 + 0.3 * u
        return ".mp4", [
            "-vf", f"scale=trunc(iw*{f:.3f}/2)*2:-2:flags=bicubic,scale={width}:{height}:flags=bicubic",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "8",
        ]
    if family == "x264mid":
        return ".mp4", ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(20 + round(u * 8))]
    if family == "vp9":
        return ".webm", [
            "-c:v", "libvpx-vp9", "-crf", str(30 + round(u * 15)), "-b:v", "0",
            "-deadline", "realtime", "-cpu-used", "8",
        ]
    if family == "denoise_sharpen":
        return ".mp4", [
            "-vf", "hqdn3d=4:3:6:4.5,unsharp=5:5:0.8", "-c:v", "libx264", "-preset", "veryfast", "-crf", "8",
        ]
    if family == "mpeg2":
        return ".mpg", ["-c:v", "mpeg2video", "-q:v", str(4 + round(u * 4))]
    raise ValueError(f"không biết họ xử lý {family}")


def video_size(ffmpeg, path):
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    found = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", probe)
    if not found:
        raise RuntimeError(f"không đọc được kích thước video {path}")
    return int(found.group(1)), int(found.group(2))


def reencode(cfg, rows):
    """Nén lại / xử lý từng mẫu theo cfg["variant"] → {sample_id: đường dẫn mp4}; có sẵn thì bỏ qua.

    - mức nén (mặc định): scale về `height`, `video_codec` (libx264 + `preset` + `crf`, hoặc
      libvpx-vp9 + `crf`) và `audio_codec` (aac / libopus) + `audio_bitrate`;
    - `kind` = "processed_real" (PR): video của clip nguồn đi qua x264 CRF 8 → bước giữa
      (processed_real_middle) → về đúng kích thước gốc, x264 CRF 18 (như đầu ra generator), một lần
      mỗi clip nguồn; rồi ghép lại nguyên tiếng của từng mẫu (tiếng không đổi, nhãn giữ nguyên)."""
    spec = cfg["variant"]
    folder = cache_roots(cfg)[0] / "clips"
    folder.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_binary()
    options = subprocess.run([ffmpeg, "-hide_banner", "-h", "full"], capture_output=True, text=True).stdout
    sync = ["-fps_mode", "passthrough"] if "fps_mode" in options else ["-vsync", "passthrough"]
    began, failed = time.perf_counter(), []

    def compress(row):
        out = folder / f"{row['sample_id']}.mp4"
        if out.is_file():
            return
        tmp = out.with_suffix(".tmp.mp4")
        codec = spec.get("video_codec", "libx264")
        video = ["-c:v", codec, "-crf", str(spec["crf"])] + (
            ["-preset", spec.get("preset", "veryfast")] if codec == "libx264"
            else ["-b:v", "0", "-deadline", "realtime", "-cpu-used", "8"]
        )
        command = [
            ffmpeg, "-y", "-v", "error", "-i", str(media_path(cfg["dataset"], row)),
            "-vf", f"scale=-2:{int(spec['height'])}", *video, "-pix_fmt", "yuv420p",
            *sync, "-c:a", spec.get("audio_codec", "aac"), "-b:a", str(spec["audio_bitrate"]), str(tmp),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            failed.append(f"{row['sample_id']}: {result.stderr[-300:]}")
            return
        os.replace(tmp, out)

    def process_real(group):
        todo = [r for r in group if not (folder / f"{r['sample_id']}.mp4").is_file()]
        if not todo:
            return
        # real / sham / voice của một clip nguồn chung dòng bit video (đo 07/11: cùng MD5).
        source = next((r for r in group if r.get("cell") == "real"), group[0])
        family, u = processed_real_chain(spec, source)
        stem = folder / f"{source['source_clip_id']}.pr"
        src = media_path(cfg["dataset"], source)
        width, height = video_size(ffmpeg, src)
        ext, middle = processed_real_middle(family, u, width, height)
        steps = [
            (src, ["-c:v", "libx264", "-preset", "veryfast", "-crf", "8", "-pix_fmt", "yuv420p"], f"{stem}.a.mp4"),
            (f"{stem}.a.mp4", middle, f"{stem}.b{ext}"),
            (f"{stem}.b{ext}", ["-vf", f"scale={width}:{height}", "-c:v", "libx264", "-preset", "medium",
                                "-crf", "18", "-pix_fmt", "yuv420p"], f"{stem}.c.mp4"),
        ]
        try:
            for inp, args, out in steps:
                result = subprocess.run(
                    [ffmpeg, "-y", "-v", "error", "-i", str(inp), "-an", *args, *sync, str(out)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"{family}: {result.stderr[-300:]}")
            for row in todo:
                out = folder / f"{row['sample_id']}.mp4"
                tmp = out.with_suffix(".tmp.mp4")
                result = subprocess.run(
                    [ffmpeg, "-y", "-v", "error", "-i", f"{stem}.c.mp4", "-i", str(media_path(cfg["dataset"], row)),
                     "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest", str(tmp)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"ghép tiếng: {result.stderr[-300:]}")
                os.replace(tmp, out)
        except Exception as exc:
            failed.append(f"{source['source_clip_id']}: {str(exc)[:300]}")
        finally:
            for leftover in folder.glob(f"{source['source_clip_id']}.pr.*"):
                leftover.unlink()

    if spec.get("kind") == "processed_real":
        groups = defaultdict(list)
        for row in rows:
            groups[row["source_clip_id"]].append(row)
        one, jobs = process_real, list(groups.values())
        plan = {
            row["sample_id"]: list(processed_real_chain(spec, next((r for r in g if r.get("cell") == "real"), g[0])))
            for g in jobs for row in g
        }
    else:
        one, jobs, plan = compress, rows, None
    workers = int(spec.get("workers") or os.cpu_count() or 1)
    with Progress(f"reencode {spec['name']}", len(jobs)) as progress, ThreadPoolExecutor(workers) as pool:
        for _ in pool.map(one, jobs):
            progress.step()
    if failed:
        raise RuntimeError(f"reencode: {len(failed)} mẫu lỗi: " + " | ".join(failed[:5]))
    write_json(
        folder / "reencode.json",
        {**spec, "samples": len(rows), "elapsed_s": time.perf_counter() - began, "ffmpeg": ffmpeg,
         **({"chains": plan} if plan else {})},
    )
    print(f"== reencode {spec['name']} xong: {len(rows)} mẫu trong {duration(time.perf_counter() - began)}", flush=True)
    return {r["sample_id"]: folder / f"{r['sample_id']}.mp4" for r in rows}


def prepare(cfg, encoders=ORDER, factory=None):
    rows, selection = training_dataset(cfg, verify_media=False, labels=False)
    cache = cache_roots(cfg)[0]
    reports = {}
    chosen = [e for e in ORDER if e in encoders]
    clips = None
    if cfg.get("variant"):
        # Biến thể nén / PR: landmark lấy của bản gốc; không trích lipinc / wave (model v005 không cần).
        chosen = [e for e in chosen if e in ("avhubert", "mouth", "xlsr")]
        rows = variant_selection(cfg["variant"], rows)
        clips = reencode(cfg, rows)
    run = tracking.open_run(
        cfg,
        "prepare",
        # 3 part cùng encoder: tên theo phiên (prepare-p<N>-<phiên>) để không trùng giữa các part.
        f"{cfg.get('session') or cache.name + '-prepare'}-{'+'.join(chosen)}",
        group=cache.name,
        config={
            "encoders": {e: cfg["encoders"][e] for e in chosen},
            "selection": selection,
            "samples": len(rows),
        },
        folder=cache,
        with_setup=True,
    )
    print(
        f"prepare: {len(rows)} mẫu, encoder {chosen} → {cache} | "
        f"thiết bị {cfg.get('prepare_devices') or ['cpu']} | {system_status()}",
        flush=True,
    )
    for index, name in enumerate(chosen, 1):
        spec = cfg["encoders"][name]
        if spec.get("cpu_only"):
            # Không dùng GPU: số worker theo số nhân CPU (Kaggle 4), kể cả trên phiên có GPU.
            pool_devices = [None] * int(spec.get("cpu_workers") or os.cpu_count() or 1)
        else:
            devices = list(cfg.get("prepare_devices") or [None])
            copies = int(spec.get("workers_per_device", 1))
            pool_devices = [d for d in devices for _ in range(copies)]
        folder = cache / name
        folder.mkdir(parents=True, exist_ok=True)
        cached = sum(1 for r in rows if (folder / f"{r['sample_id']}.json").is_file())
        if name in ("avhubert", "mouth", "lipinc"):
            ready = sum(cache_file(cfg, "landmarks", r["sample_id"]).is_file() for r in rows)
            print(
                f"{name}: landmark có sẵn (stage landmarks) cho {ready}/{len(rows)} mẫu"
                + ("" if ready == len(rows) else "; mẫu còn lại dò bằng dlib / đọc cache avhubert"),
                flush=True,
            )
        print(
            f"== [{index}/{len(chosen)}] {name}: {len(rows)} mẫu, đã có {cached} file cache "
            f"(khớp chữ ký thì bỏ qua) | {len(pool_devices)} worker trên {pool_devices}",
            flush=True,
        )
        began = time.perf_counter()
        available = queue.Queue()
        with Progress(f"{name}: tạo worker", len(pool_devices), unit="worker", every_n=1) as made:
            for device in pool_devices:
                encoder = (factory or make)(cfg, name, device)
                getattr(encoder, "start", lambda: None)()  # nạp model lần lượt, không cùng lúc
                available.put(encoder)
                made.step()
        startup = time.perf_counter() - began
        groups = defaultdict(list)
        for row in rows:
            groups[row.get("source_clip_id") or row["sample_id"]].append(row)
        guard, done, first, failed = threading.Lock(), [0], [None], []

        def run_row(encoder, row):
            video = clips[row["sample_id"]] if clips else media_path(cfg["dataset"], row)
            request = {"video": str(video)}
            if name in ("mouth", "lipinc"):
                request["box_cache"] = str(landmark_file(cfg, row["sample_id"]))
            elif name == "avhubert" and cache_file(cfg, "landmarks", row["sample_id"]).is_file():
                request["landmarks"] = str(cache_file(cfg, "landmarks", row["sample_id"]))
            encoder.extract(request, folder / f"{row['sample_id']}.npz")

        def extract(group):
            encoder = available.get()
            try:
                for row in group:
                    try:
                        run_row(encoder, row)
                    except Exception as exc:
                        # Một mẫu lỗi (vd. worker bị giết vì hết RAM) không bỏ cả nhóm: ghi lại, thử
                        # lại ở cuối; worker chết thì lần gọi sau tự mở lại.
                        print(
                            f"{name}: LỖI ở {row['sample_id']}, thử lại ở cuối: {str(exc)[:600]}",
                            flush=True,
                        )
                        with guard:
                            failed.append(row)
                        continue
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
            if failed:
                print(f"{name}: thử lại {len(failed)} mẫu lỗi, lần lượt", flush=True)
                encoder, still = available.get(), []
                try:
                    for row in failed:
                        try:
                            run_row(encoder, row)
                        except Exception as exc:
                            still.append(f"{row['sample_id']}: {str(exc)[:300]}")
                finally:
                    available.put(encoder)
                if still:
                    raise RuntimeError(
                        f"{name}: {len(still)} mẫu vẫn lỗi sau khi thử lại (chạy lại phiên với "
                        f"PREVIOUS_CACHE để trích tiếp): " + " | ".join(still[:5])
                    )
                print(f"{name}: thử lại xong, {len(failed)} mẫu đều được", flush=True)
        finally:
            while not available.empty():
                getattr(available.get(), "close", lambda: None)()
        elapsed = time.perf_counter() - began
        quality = {**cache_quality(name, folder, rows), **log_cache(run, name, folder, rows)}
        if name == "avhubert":
            # v004: đường cong khớp theo độ trễ cho mọi mẫu (thước AV-HuBERT của nhãn mismatch đo
            # được, training/sync_labels.py); rẻ, chỉ đọc lại cache vừa trích.
            from vn_av_df.training.sync_labels import avhubert_profiles

            quality["sync_profile"] = avhubert_profiles(cfg, rows, folder)
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
    if name not in ("landmarks", "avhubert", "mouth"):
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
            elif name == "landmarks":
                valid.append(float(z["visual_valid"].mean()))
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
