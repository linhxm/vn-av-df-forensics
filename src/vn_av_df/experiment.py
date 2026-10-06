"""Shared protocol: cache once, train architectures, calibrate on validation, test once."""

import copy
import io
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from vn_av_df import tracking
from vn_av_df.common.runtime import atomic_bytes, fingerprint, read_json, sha, write_json
from vn_av_df.dataset import (
    condition,
    media_path,
    protocol_rows,
    temporal_targets,
    training_dataset,
)
from vn_av_df.features.registry import (
    artifact_arrays,
    combined_signature,
    feature_arrays,
    make_encoder,
    method_config,
)
from vn_av_df.metrics import binary_metrics, choose_threshold, intervals, temporal_ap
from vn_av_df.models import build_model, output_valid, pool_score

FORMAT = "vn-av-df-temporal-v1"


def prediction_records(predictions):
    """Lưu đủ nhãn/timeline để vẽ báo cáo hoặc so sánh offline từ ZIP, không cần video."""
    return [
        dict(
            sample_id=p["row"]["sample_id"],
            label=p["row"]["label"],
            split=p["row"]["split"],
            condition=condition(p["row"]),
            dataset_part=p["row"].get("dataset_part"),
            source_id=p["row"]["source_id"],
            duration_s=p["row"]["duration_s"],
            fake_intervals=p["row"]["fake_intervals"],
            av_mismatch_intervals=p["row"].get("av_mismatch_intervals"),
            video_score=p["video_score"],
            times_s=p["times"].tolist(),
            scores=p["scores"].tolist(),
            valid=p["valid"].tolist(),
            branch_scores={name: value.tolist() for name, value in p["branches"].items()},
        )
        for p in predictions
    ]


def save_checkpoint(path, state):
    stream = io.BytesIO()
    torch.save(state, stream)
    atomic_bytes(path, stream.getvalue())


ENCODER_CACHES = ("fate", "avhubert", "dinov2")  # Thứ tự trích: DINOv2 cần hộp miệng AV-HuBERT.


def attach_caches(sources, target, writable=()):
    """Gom cache encoder từ nhiều thư mục (Kaggle input chỉ đọc) vào `target/<encoder>`.

    Encoder trong `writable` được chép để trích tiếp; encoder khác chỉ tạo symlink (chỉ đọc).
    Trả về {encoder: thư mục nguồn}.
    """
    import shutil

    found = {}
    for source in map(Path, sources):
        for key in ENCODER_CACHES:
            folder = source / key
            if folder.is_dir():
                if key in found:
                    raise ValueError(f"Cache {key} có ở cả {found[key]} và {folder}")
                found[key] = folder
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for key, folder in found.items():
        destination = target / key
        if destination.is_symlink() or destination.exists():
            continue  # Chạy lại cell: giữ bản đã gom/đang trích.
        if key in writable:
            shutil.copytree(folder, destination)
        else:
            destination.symlink_to(folder.resolve(), target_is_directory=True)
    return found


def prepare(cfg):
    """Trích mỗi backbone một lần; test chỉ được extract, không fit thống kê.

    cfg["prepare_encoders"] (vd. ["dinov2"]) giới hạn encoder được trích; None = mọi encoder
    mà các kiến trúc đã chọn cần.
    """
    if cfg.get("methods"):
        reports, seen = {}, set()
        only = cfg.get("prepare_encoders")
        for name in cfg["architectures"]:
            spec = cfg["methods"][name]
            scoped = method_config(cfg, name)
            artifact = {
                "encoder": scoped.pop("artifact_encoder", None),
                "cache": scoped.pop("artifact_cache", None),
            }
            scoped.pop("methods", None)
            # Artifact P2 đọc hộp miệng từ cache encoder chính nên luôn chạy sau nó.
            stages = [(spec["encoder"], scoped)]
            if artifact["encoder"]:
                stages.append((spec["artifact_encoder"], {**scoped, **artifact}))
            for key, stage in stages:
                if key not in seen and (only is None or key in only):
                    reports[key] = prepare(stage)
                    seen.add(key)
        return reports
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    rows, selection = training_dataset(cfg, verify_media=True)
    # Mỗi device (vd. cuda:0, cuda:1) workers_per_device encoder, mẫu chia động giữa các encoder.
    # Device không nằm trong feature signature nên cache giống nhau dù trích ở GPU nào.
    # workers_per_device > 1 khi phần chậm chạy CPU một luồng (dlib của AV-HuBERT).
    copies = int(cfg["encoder"].get("workers_per_device", 1))
    devices = [d for d in (cfg.get("prepare_devices") or [None]) for _ in range(copies)]
    encoders = queue.Queue()
    for device in devices:
        encoders.put(
            make_encoder(cfg["encoder"] if device is None else {**cfg["encoder"], "device": device})
        )
    name = Path(cfg["cache"]).name
    print(
        f"Prepare {name}: {len(rows)} mẫu, cache {cfg['cache']}, device {devices}",
        flush=True,
    )
    start = time.perf_counter()
    guard, progress = threading.Lock(), {"done": 0}

    def extract(group):
        # Cả nhóm cùng parent qua một encoder: real/sham/partial có frame giống nhau, encoder
        # dùng lại được kết quả theo frame (landmark AV-HuBERT).
        encoder = encoders.get()
        try:
            for row in group:
                tick = time.perf_counter()
                encoder.extract(
                    {"video": str(media_path(cfg["dataset"], row))},
                    Path(cfg["cache"]) / (row["sample_id"] + ".npz"),
                )
                with guard:
                    progress["done"] += 1
                    done, spent = progress["done"], time.perf_counter() - start
                # Mẫu đã có cache mất ~0 s; thời gian còn lại ước theo tốc độ trung bình đến giờ.
                print(
                    f"Features {name} {done}/{len(rows)} {row['sample_id']}: "
                    f"{time.perf_counter() - tick:.1f}s | đã chạy {spent / 60:.1f} phút, "
                    f"còn ~{spent / done * (len(rows) - done) / 60:.1f} phút",
                    flush=True,
                )
        finally:
            encoders.put(encoder)

    groups = defaultdict(list)
    for row in rows:
        groups[row.get("source_clip_id") or row["sample_id"]].append(row)
    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(extract, group) for group in groups.values()]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                # Một mẫu lỗi: không nhận nhóm mới, chờ nhóm đang chạy xong rồi báo lỗi.
                for future in futures:
                    future.cancel()
                raise
    finally:
        signature = encoders.queue[0].signature
        while not encoders.empty():  # Đóng worker riêng (AV-HuBERT) nếu encoder có.
            getattr(encoders.get(), "close", lambda: None)()
    elapsed = time.perf_counter() - start
    write_json(
        Path(cfg["cache"]) / "extraction.json",
        {
            "samples": len(rows),
            "dataset_selection": selection,
            "elapsed_s": elapsed,
            "device": cfg["encoder"]["device"],
            "devices": devices,
            "feature_signature": signature,
            "note": "Includes loading, decoding and reusable-cache checks",
        },
    )
    return {"samples": len(rows), "elapsed_s": elapsed}


def cached(row, cfg, device):
    """Kiểm hash rồi chuyển native features thành input và grid output của model."""
    path = Path(cfg["cache"]) / (row["sample_id"] + ".npz")
    if not path.with_suffix(".json").is_file():
        raise FileNotFoundError(
            f"Thiếu feature cache {path.with_suffix('.json')}: bước prepare chưa chạy xong "
            "(lỗi/bị dừng giữa chừng hoặc khác dữ liệu/encoder). Chạy lại prepare đến hết rồi train."
        )
    meta = read_json(path.with_suffix(".json"))
    expected = fingerprint({"assets": {"video": row["sha256"]}, "variant": {"kind": "clean"}})
    if meta["feature_sha256"] != sha(path) or meta["source_fingerprint"] != expected:
        raise ValueError("Stale/corrupt feature cache or media provenance")
    (audio, visual, valid), times, meta = feature_arrays(
        path, meta, cfg.get("model_architecture", "gru")
    )
    if (
        not len(times)
        or not np.allclose(np.diff(times), meta["step_s"])
        or times[0] != 0
        or abs(meta["duration_s"] - row["duration_s"]) > 0.1
    ):
        raise ValueError("Cache timeline differs from labeled video")
    arrays = [audio, visual, valid]
    if cfg.get("artifact_cache"):
        # P2: DINOv2 miệng phải thuộc đúng video và đúng cache hộp miệng AV-HuBERT.
        extra = Path(cfg["artifact_cache"]) / (row["sample_id"] + ".npz")
        extra_meta = read_json(extra.with_suffix(".json"))
        if (
            extra_meta["feature_sha256"] != sha(extra)
            or extra_meta["source_fingerprint"] != expected
            or extra_meta["box_sha256"] != meta["feature_sha256"]
        ):
            raise ValueError("Stale/corrupt artifact cache or media provenance")
        artifact, artifact_valid = artifact_arrays(extra, extra_meta, len(audio))
        arrays = [audio, visual, valid & artifact_valid, artifact]
        meta = {
            **meta,
            "feature_signature": combined_signature(
                meta["feature_signature"], extra_meta["feature_signature"]
            ),
        }
    tensors = [torch.as_tensor(x, device=device) for x in arrays]
    target = torch.as_tensor(temporal_targets(row, times, meta["step_s"]), device=device)
    return tensors, target, times, meta


def objective(logits, valid, target, row, top_fraction):
    observed = valid & (target >= 0)
    loss = (
        F.binary_cross_entropy_with_logits(logits[observed], target[observed])
        if observed.any()
        else None
    )
    # Strong labels: no clip-positive loss if its manipulated region is entirely unobservable.
    video_observed = bool(valid.any()) and (
        row["label"] == 0 or row["fake_intervals"] is None or bool((valid & (target > 0)).any())
    )
    if video_observed:
        score = pool_score(logits.sigmoid(), valid, top_fraction)
        video_loss = F.binary_cross_entropy(
            score.clamp(1e-6, 1 - 1e-6), score.new_tensor(float(row["label"]))
        )
        loss = video_loss if loss is None else loss + video_loss
    return loss


def training_loss(model, logits, valid, target, row, options):
    """Loss AI; P2 cộng head phụ artifact theo ô để timeline artifact có nghĩa riêng."""
    loss = objective(logits, valid, target, row, options["top_fraction"])
    artifact = getattr(model, "outputs", {}).get("artifact")
    weight = options.get("artifact_aux_weight", 0.5)
    observed = valid & (target >= 0)
    if artifact is not None and weight and observed.any():
        extra = weight * F.binary_cross_entropy_with_logits(artifact[observed], target[observed])
        loss = extra if loss is None else loss + extra
    return loss


def branch_reports(groups, top_fraction):
    """AUC không ngưỡng của từng nhánh P2: real gốc (0) so với từng ô 2×2 (1)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    names = sorted({k for items in groups.values() for p in items for k in p["branches"]})
    if not names:
        return None

    def pooled(p, name):
        values = p["branches"][name][p["valid"]]
        if not len(values):
            return None
        return float(np.sort(values)[-max(1, math.ceil(len(values) * top_fraction)) :].mean())

    result = {}
    for name in names:
        result[name] = {}
        for cell, items in sorted(groups.items()):
            if cell == "real":
                continue
            pairs = [(0, pooled(p, name)) for p in groups.get("real", [])]
            pairs += [(1, pooled(p, name)) for p in items]
            kept = [(y, s) for y, s in pairs if s is not None]
            labels, scores = [y for y, _ in kept], [s for _, s in kept]
            both = len(set(labels)) == 2
            result[name][cell] = {
                "samples": len(labels),
                "roc_auc": float(roc_auc_score(labels, scores)) if both else None,
                "average_precision": float(average_precision_score(labels, scores))
                if both
                else None,
            }
    return {
        "top_fraction": top_fraction,
        "scores": result,
        "note": "Positive = condition vs original real; sham measures A/V-mismatch sensitivity",
    }


def breakdown(predictions, thresholds, top_fraction):
    """Theo ô 2×2 (real gốc tách khỏi sham) và theo nhánh bằng chứng nếu model có."""
    groups = defaultdict(list)
    for p in predictions:
        groups[condition(p["row"])].append(p)
    reals = groups.get("real", [])
    by_condition = {
        cell: summarize(items if cell in ("real", "sham") else reals + items, thresholds)
        for cell, items in sorted(groups.items())
    }
    return by_condition, branch_reports(groups, top_fraction)


@torch.no_grad()
def predict_rows(model, rows, cfg):
    model.eval()
    results = []
    losses = []
    device = cfg["device"]
    for row in rows:
        tensors, target, times, meta = cached(row, cfg, device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(*tensors)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        valid = output_valid(model, tensors[2])
        loss = training_loss(model, logits, valid, target, row, cfg["training"])
        if loss is not None:
            losses.append(float(loss))
        score = pool_score(logits.sigmoid(), valid, cfg["training"]["top_fraction"])
        results.append(
            {
                "row": row,
                "scores": logits.sigmoid().cpu().numpy(),
                "valid": valid.cpu().numpy(),
                "target": target.cpu().numpy(),
                "times": times,
                "step": meta["step_s"],
                "video_score": float(score) if score is not None else None,
                "inference_s": elapsed,
                "branches": {
                    name: value.sigmoid().cpu().numpy()
                    for name, value in getattr(model, "outputs", {}).items()
                    if value is not None
                },
            }
        )
    return results, float(np.mean(losses)) if losses else None


def calibrate(results):
    measured = [r for r in results if r["video_score"] is not None]
    video_threshold = choose_threshold(
        [r["row"]["label"] for r in measured], [r["video_score"] for r in measured]
    )
    labels = []
    scores = []
    for r in results:
        # A small edited fraction is positive for localization; no rounding short events away.
        mask = r["valid"] & (r["target"] >= 0)
        labels.extend((r["target"][mask] > 0).astype(int).tolist())
        scores.extend(r["scores"][mask].tolist())
    temporal_threshold = choose_threshold(labels, scores) if set(labels) == {0, 1} else None
    return {"video": video_threshold, "temporal": temporal_threshold}


def summarize(results, thresholds):
    measured = [r for r in results if r["video_score"] is not None]
    video = binary_metrics(
        [r["row"]["label"] for r in measured],
        [r["video_score"] for r in measured],
        thresholds["video"],
    )
    prediction = {}
    truth = {}
    real_alerts = []
    real_events, real_observed_seconds, real_input_seconds = 0, 0.0, 0.0
    known_cells = 0
    valid_cells = 0
    for r in results:
        row = r["row"]
        sid = row["sample_id"]
        spans = (
            intervals(
                r["scores"],
                r["valid"],
                r["times"],
                r["step"],
                row["duration_s"],
                thresholds["temporal"],
            )
            if thresholds["temporal"] is not None
            else []
        )
        prediction[sid] = spans
        if row["fake_intervals"] is not None:
            truth[sid] = row["fake_intervals"]
        if row["label"] == 0:
            real_alerts.append(bool(spans))
            real_events += len(spans)
            real_input_seconds += row["duration_s"]
            widths = np.minimum(r["times"] + r["step"], row["duration_s"]) - r["times"]
            real_observed_seconds += float(widths[r["valid"]].sum())
        known_cells += len(r["valid"])
        valid_cells += int(r["valid"].sum())
    temporal = (
        {str(t): temporal_ap(prediction, truth, t) for t in (0.3, 0.5, 0.75, 0.95)}
        if thresholds["temporal"] is not None
        else None
    )
    return {
        "video": video,
        "temporal": temporal,
        "temporal_labeled_videos": len(truth),
        "video_coverage": len(measured) / max(1, len(results)),
        "window_coverage": valid_cells / max(1, known_cells),
        "real_video_interval_false_alarm_rate": float(np.mean(real_alerts))
        if real_alerts and temporal is not None
        else None,
        "head_inference_seconds": sum(r["inference_s"] for r in results),
        "false_intervals_per_real_hour": real_events * 3600 / real_observed_seconds
        if real_observed_seconds and temporal is not None
        else None,
        "real_observed_seconds": real_observed_seconds,
        "real_input_seconds": real_input_seconds,
        "latency_note": "Temporal detector only; feature extraction cost recorded separately",
    }


def train_one(cfg, architecture, seed, resume=False):
    """Một method/seed: real reconstruction (nếu P1), supervised head, validation."""
    cfg = method_config(cfg, architecture)
    head = cfg["model_architecture"]
    torch.set_num_threads(cfg["training"].get("torch_threads", 2))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if str(cfg["device"]).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    rows, selection = training_dataset(cfg)
    rows = protocol_rows(rows, cfg.get("held_out_generators", []))
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    for name, subset in (("train", train), ("validation", val)):
        if {r["label"] for r in subset} != {0, 1}:
            raise ValueError(f"Need real and fake in {name}")
    signatures = set()
    cache_hashes = []
    first = None
    for r in train + val:
        tensors, _, _, meta = cached(r, cfg, "cpu")
        signatures.add(meta["feature_signature"])
        cache_hashes.append(meta["feature_sha256"])
        first = tensors if first is None else first
    if len(signatures) != 1:
        raise ValueError("Mixed encoder versions in cache")
    options = cfg["training"]
    if options["epochs"] < 1:
        raise ValueError("epochs must be positive")
    if not 0 < options["top_fraction"] <= 1:
        raise ValueError("top_fraction must be in (0,1]")
    model = build_model(
        first[0].shape[-1],
        first[1].shape[-1],
        head,
        options["hidden"],
        options["dropout"],
        native_stride=meta.get("native_stride", 1),
        **({"artifact_dim": first[3].shape[-1]} if len(first) > 3 else {}),
    ).to(cfg["device"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=options["lr"],
        weight_decay=options["weight_decay"],
    )
    out = Path(cfg["runs"]) / f"{architecture}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    run_id = fingerprint(
        {
            "rows": train + val,
            "manifest_sha256": selection["manifest_sha256"],
            "cache": cache_hashes,
            "options": {k: v for k, v in options.items() if k != "epochs"},
            "model": model.config,
            "seed": seed,
            "reconstruction": cfg.get("reconstruction", {}),
            "sync": cfg.get("sync", {}),
            "implementation": {
                name: fingerprint((Path(__file__).parent / name).read_text(encoding="utf-8"))
                for name in (
                    "experiment.py",
                    "models.py",
                    "reconstruction.py",
                    "syncartifact.py",
                    "dataset.py",
                    "metrics.py",
                    "features/registry.py",
                )
            },
        }
    )
    start = 0
    best = float("inf")
    stale = 0
    history = []
    reconstruction_report = sync_report = None
    if (out / "last.pt").exists() and not resume:
        raise FileExistsError("Run exists: set RESUME=True or choose RUN_NAME")
    # Số mẫu theo split/ô 2×2 (test chỉ đếm, không đọc) để biết detector train trên dữ liệu nào.
    data = dict(sorted(Counter(f"{r['split']}/{condition(r)}" for r in rows).items()))
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Train {out.name}: {len(train)} train, {len(val)} validation, "
        f"{parameters:,} tham số train được; {data}",
        flush=True,
    )
    tracking.start(cfg, out, architecture, seed, data)  # train_selected đóng run kể cả khi lỗi.
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if state["run_id"] != run_id:
            raise ValueError("Resume configuration/data/code changed")
        model.load_state_dict(state["state"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["epoch"] + 1
        best = state["best"]
        stale = state["stale"]
        history = state["history"]
        reconstruction_report = state.get("reconstruction_report")
        sync_report = state.get("sync_report")
        # Run W&B mới của phiên resume: ghi lại lịch sử đã có trước khi train tiếp.
        for stage, rows in (
            ("stageA", (reconstruction_report or {}).get("history", [])),
            ("stageS", (sync_report or {}).get("history", [])),
            ("detector", history),
        ):
            for row in rows:
                tracking.log(stage, row, echo=False)
        torch.set_rng_state(state["rng"])
        if str(cfg["device"]).startswith("cuda") and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if options["patience"] and stale >= options["patience"]:
            start = options["epochs"]  # Đã early stop: resume không train thêm, giữ best.pt.
    elif hasattr(model, "reconstructor"):
        from vn_av_df.reconstruction import fit_reconstruction

        def load(row):
            return cached(row, cfg, cfg["device"])

        # artifact_only của P2 không dùng R/nhánh sync nên đã sẵn sàng từ đầu.
        if not bool(model.reconstruction_ready):
            reconstruction_report = fit_reconstruction(
                model,
                train,
                val,
                lambda row: load(row)[0],
                cfg.get("reconstruction", options),
                seed,
            )
            write_json(out / "reconstruction.json", reconstruction_report)
        if hasattr(model, "sync_ready") and not bool(model.sync_ready):
            from vn_av_df.syncartifact import fit_sync

            sync_report = fit_sync(
                model, train, val, load, cfg.get("sync", {}), seed, options["top_fraction"]
            )
            write_json(out / "sync.json", sync_report)
    began = time.perf_counter()
    write_json(
        out / "configuration.json",
        {"run_id": run_id, "architecture": architecture, "seed": seed, "settings": cfg},
    )
    for epoch in range(start, options["epochs"]):
        tick = time.perf_counter()
        model.train()
        if options.get("balance_parents", True):
            from vn_av_df.reconstruction import balanced_order

            order = balanced_order(train, seed + epoch)
        else:
            order = train.copy()
            random.Random(seed + epoch).shuffle(order)
        losses = []
        for row in order:
            tensors, target, _, _ = cached(row, cfg, cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(*tensors)
            loss = training_loss(
                model, logits, output_valid(model, tensors[2]), target, row, options
            )
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:
            raise ValueError("No usable training windows")
        prediction, val_loss = predict_rows(model, val, cfg)
        if val_loss is None:
            raise ValueError("No usable validation windows")
        improved = val_loss < best
        best, stale = (val_loss, 0) if improved else (best, stale + 1)
        thresholds = calibrate(prediction)
        report = summarize(prediction, thresholds)
        report["by_condition"], report["branches"] = breakdown(
            prediction, thresholds, options["top_fraction"]
        )
        # Theo ô 2×2: AUC fake so với real gốc; real/sham chỉ có false alarm (một lớp).
        cells = {}
        for cell, metrics in report["by_condition"].items():
            key = "auc" if cell not in ("real", "sham") else "false_alarm"
            cells[f"validation_{key}/{cell}"] = metrics["video"][
                "roc_auc" if key == "auc" else "false_alarm_rate"
            ]
        # P2: AUC không ngưỡng của từng nhánh theo ô; theo dõi nhánh nào học được gì qua epoch.
        for branch, values in ((report["branches"] or {}).get("scores") or {}).items():
            for cell, metrics in values.items():
                cells[f"branch_auc/{branch}/{cell}"] = metrics["roc_auc"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_loss": val_loss,
                "validation_roc_auc": report["video"]["roc_auc"],
                "validation_f1": report["video"]["f1"],
                "validation_coverage": report["video_coverage"],
                "validation_pr_auc": report["video"]["pr_auc"],
                "validation_false_alarm_rate": report["video"]["false_alarm_rate"],
                "validation_temporal_ap_0.5": ((report["temporal"] or {}).get("0.5") or {}).get(
                    "ap"
                ),
                "video_threshold": thresholds["video"],
                "best_validation_loss": best,
                "epochs_without_improvement": stale,
                "epoch_seconds": time.perf_counter() - tick,
                **cells,
            }
        )
        tracking.log("detector", history[-1])
        state = {
            "format": FORMAT,
            "state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model.config,
            "feature_signature": next(iter(signatures)),
            "encoder_config": cfg["encoder"],
            "artifact_encoder_config": cfg.get("artifact_encoder"),
            "method_name": architecture,
            "reconstruction_report": reconstruction_report,
            "sync_report": sync_report,
            "thresholds": thresholds,
            "top_fraction": options["top_fraction"],
            "epoch": epoch,
            "best": best,
            "stale": stale,
            "history": history,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if str(cfg["device"]).startswith("cuda")
            else None,
            "run_id": run_id,
            "seed": seed,
            "held_out_generators": cfg.get("held_out_generators", []),
            "manifest_sha256": selection["manifest_sha256"],
            "dataset_selection": selection,
            "validation_report": report,
        }
        save_checkpoint(out / "last.pt", state)
        if improved:
            save_checkpoint(out / "best.pt", state)
            write_json(out / "validation.json", report)
            write_json(out / "predictions-validation.json", prediction_records(prediction))
            write_json(
                out / "best.json",
                {
                    "epoch": epoch + 1,
                    "validation_loss": val_loss,
                    "thresholds": thresholds,
                    "selection": "minimum validation loss",
                },
            )
        write_json(out / "history.json", history)
        if improved:
            print(f"{out.name}: best mới ở epoch {epoch + 1}, lưu best.pt", flush=True)
        if options["patience"] and stale >= options["patience"]:
            print(f"{out.name}: early stop sau {stale} epoch không cải thiện", flush=True)
            break
    resources = {
        "elapsed_this_session_s": time.perf_counter() - began,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_head_parameters": sum(p.numel() for p in model.parameters()),
        "reconstruction_seconds": reconstruction_report["elapsed_s"]
        if reconstruction_report
        else 0,
        "sync_seconds": sync_report["elapsed_s"] if sync_report else 0,
        "device": cfg["device"],
        "peak_gpu_bytes": torch.cuda.max_memory_allocated()
        if str(cfg["device"]).startswith("cuda")
        else None,
    }
    write_json(out / "resources.json", resources)
    tracking.finish(
        {
            "best_epoch": read_json(out / "best.json")["epoch"],
            **{f"resources/{k}": v for k, v in resources.items()},
        }
    )
    return out / "best.pt"


def load_model(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT:
        raise ValueError(
            "Need a new binary temporal checkpoint; relation-head checkpoints are incompatible"
        )
    model = build_model(**state["model_config"]).to(device)
    model.load_state_dict(state["state"])
    model.eval()
    return model, state


def resolve_checkpoint(cfg, recorded):
    path = Path(recorded)
    if path.is_file():
        return path
    # Artifact ZIP preserves <architecture_seed>/best.pt beneath the configured run.
    parts = str(recorded).replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[-1] in {"best.pt", "last.pt"}:
        candidate = Path(cfg["runs"]) / parts[-2] / parts[-1]
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Checkpoint not found after artifact relocation: {recorded}")


def train_selected(cfg, resume=False):
    """Train đúng danh sách model/seed, lưu từng kết quả; không xếp hạng hoặc chạy test."""
    _, selection = training_dataset(cfg)
    if not cfg["architectures"] or not cfg["seeds"]:
        raise ValueError("Configure at least one architecture and seed")
    if len(set(cfg["architectures"])) != len(cfg["architectures"]) or len(set(cfg["seeds"])) != len(
        cfg["seeds"]
    ):
        raise ValueError("Architectures and seeds must not contain duplicates")
    if cfg.get("methods") and set(cfg["architectures"]) - set(cfg["methods"]):
        raise ValueError("Unknown architecture in selection")
    index_path = Path(cfg["runs"]) / "training.json"
    requested = {
        "architectures": cfg["architectures"],
        "seeds": cfg["seeds"],
        "dataset_selection": selection,
    }
    if index_path.exists():
        previous = read_json(index_path)
        if any(previous.get(k) != v for k, v in requested.items()):
            raise ValueError("Training selection changed; use a new RUN_NAME")
    # Ghi sau từng detector để phần hoàn thành vẫn báo cáo được nếu phiên bị ngắt.
    report = {**requested, "runs": [], "status": "training"}
    results = []
    for architecture in cfg["architectures"]:
        for seed in cfg["seeds"]:
            try:
                checkpoint = train_one(cfg, architecture, seed, resume)
            finally:
                tracking.finish()  # Không để run W&B mở khi một detector lỗi.
            _, state = load_model(checkpoint)
            if state["manifest_sha256"] != selection["manifest_sha256"]:
                raise ValueError("Dataset parts changed during training")
            results.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "validation": state["validation_report"],
                    "resources": read_json(checkpoint.parent / "resources.json"),
                }
            )
            report["runs"] = results
            write_json(index_path, report)
    report["status"] = "complete"
    write_json(index_path, report)
    write_json(Path(cfg["runs"]) / "dataset-selection.json", selection)
    return report


def compare(cfg, resume=False):
    """API so sánh tường minh giữ cho thí nghiệm cũ; action train không gọi hàm này."""
    report = train_selected(cfg, resume)
    results = report["runs"]
    # Architecture chosen from mean validation AUC across seeds, not test or best lucky seed.
    scores = {
        name: float(
            np.mean(
                [r["validation"]["video"]["roc_auc"] for r in results if r["architecture"] == name]
            )
        )
        for name in cfg["architectures"]
    }
    selected = max(scores, key=scores.get)
    chosen = next(
        r for r in results if r["architecture"] == selected and r["seed"] == cfg["seeds"][0]
    )
    report = {
        "dataset_selection": report["dataset_selection"],
        "runs": results,
        "mean_validation_auc": scores,
        "selected_architecture": selected,
        "demo_checkpoint": chosen["checkpoint"],
        "selection": "mean validation ROC-AUC across seeds; demo uses first configured seed, never test",
    }
    write_json(Path(cfg["runs"]) / "comparison.json", report)
    return report


def evaluate(cfg):
    """Test với checkpoint/threshold đã khóa; chạy lại cùng protocol không đổi quyết định."""
    folder = Path(cfg["runs"])
    legacy_compare = (folder / "comparison.json").is_file()
    index_path = folder / ("comparison.json" if legacy_compare else "training.json")
    comparison = read_json(index_path)
    if not legacy_compare and comparison.get("status") != "complete":
        raise ValueError("Finish the selected training runs before test evaluation")
    all_rows, selection = training_dataset(cfg)
    lock_path = Path(cfg["runs"]) / "test-lock.json"
    lock = {
        "manifest_sha256": selection["manifest_sha256"],
        "run_index_sha256": sha(index_path),
        "checkpoints": [sha(resolve_checkpoint(cfg, r["checkpoint"])) for r in comparison["runs"]],
    }
    if lock_path.exists() and read_json(lock_path) != lock:
        raise ValueError("Test protocol changed after first evaluation; use a new experiment")
    write_json(lock_path, lock)
    reports = []
    all_predictions = {}
    for run in comparison["runs"]:
        scoped = method_config(cfg, run["architecture"])
        checkpoint = resolve_checkpoint(scoped, run["checkpoint"])
        model, state = load_model(checkpoint, cfg["device"])
        if state["manifest_sha256"] != selection["manifest_sha256"]:
            raise ValueError("Evaluation dataset differs from trained experiment")
        rows = [r for r in all_rows if r["split"] == "test"]
        for row in rows:
            _, _, _, meta = cached(row, scoped, "cpu")
            if meta["feature_signature"] != state["feature_signature"]:
                raise ValueError("Test encoder differs")
        eval_cfg = copy.deepcopy(scoped)
        eval_cfg["training"]["top_fraction"] = state["top_fraction"]
        predictions, _ = predict_rows(model, rows, eval_cfg)
        all_predictions[(run["architecture"], run["seed"])] = predictions
        report = summarize(predictions, state["thresholds"])
        report["by_condition"], report["branches"] = breakdown(
            predictions, state["thresholds"], state["top_fraction"]
        )
        report["by_part"] = {
            part: summarize(
                [p for p in predictions if p["row"].get("dataset_part") == part],
                state["thresholds"],
            )
            for part in sorted({r["dataset_part"] for r in rows if "dataset_part" in r})
        }
        from vn_av_df.metrics import grouped_auc_interval

        report["uncertainty"] = grouped_auc_interval(
            predictions, cfg.get("bootstrap_replicates", 1000)
        )
        report["by_variant"] = {
            variant: summarize(
                [
                    r
                    for r in predictions
                    if r["row"]["label"] == 0 or r["row"].get("variant") == variant
                ],
                state["thresholds"],
            )
            for variant in sorted(
                {r["row"].get("variant", "unknown") for r in predictions if r["row"]["label"] == 1}
            )
        }
        write_json(
            checkpoint.parent / "predictions-test.json",
            prediction_records(predictions),
        )
        generators = sorted({r["generator"] for r in rows if r["label"] == 1})
        report["by_generator"] = {
            g: summarize(
                [r for r in predictions if r["row"]["label"] == 0 or r["row"]["generator"] == g],
                state["thresholds"],
            )
            for g in generators
        }
        report.update(
            checkpoint_sha256=sha(checkpoint),
            held_out_generators=state["held_out_generators"],
            thresholds=state["thresholds"],
        )
        write_json(checkpoint.parent / "test.json", report)
        video = report["video"]
        print(
            f"Test {checkpoint.parent.name}: {video['samples']} video, ROC-AUC "
            f"{video['roc_auc']}, F1 {video['f1']}; theo generator "
            f"{ {g: m['video']['roc_auc'] for g, m in report['by_generator'].items()} }",
            flush=True,
        )
        reports.append({"architecture": run["architecture"], "seed": run["seed"], **report})
    if not legacy_compare:
        # Chỉ là danh mục báo cáo từng detector; không tính xếp hạng/paired comparisons.
        write_json(folder / "evaluation.json", {"dataset_selection": selection, "runs": reports})
        return reports
    write_json(Path(cfg["runs"]) / "test-comparison.json", reports)
    from vn_av_df.metrics import paired_auc_interval

    summary = {}
    for name in cfg["architectures"]:
        values = [
            r["video"]["roc_auc"]
            for r in reports
            if r["architecture"] == name and r["video"]["roc_auc"] is not None
        ]
        summary[name] = {
            "seeds_measured": len(values),
            "roc_auc_mean": float(np.mean(values)) if values else None,
            "roc_auc_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    paired = {}
    for index, first in enumerate(cfg["architectures"]):
        for second in cfg["architectures"][index + 1 :]:
            for seed in cfg["seeds"]:
                paired[f"{first}-minus-{second}/seed{seed}"] = paired_auc_interval(
                    all_predictions[(first, seed)],
                    all_predictions[(second, seed)],
                    cfg.get("bootstrap_replicates", 1000),
                )
    write_json(
        Path(cfg["runs"]) / "test-summary.json", {"across_seeds": summary, "paired_auc": paired}
    )
    return reports
