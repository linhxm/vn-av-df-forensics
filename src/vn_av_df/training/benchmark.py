"""Đo thời gian và bộ nhớ train của từng model trước khi chạy thật, để ước số giờ GPU mỗi part.

Hai chế độ:
- thật (mặc định): nạp batch từ cache đã trích bằng đúng bộ nạp của train (gồm đọc ổ đĩa, augment,
  pseudo-fake, RawBoost) → thời gian sát thực tế nhất;
- tổng hợp (`synthetic=True`): tensor ngẫu nhiên đúng kích thước thật trên GPU, không cần dữ liệu
  hay cache → chỉ đo phần tính toán (dùng được trên máy mới trước khi có dữ liệu). XLS-R lấy từ
  `model_assets["xlsr"]` nếu đã tải, không thì dựng đúng cấu hình 300M với trọng số ngẫu nhiên.

Mỗi model: `steps` bước train (bỏ `warmup` bước đầu) với đúng công thức train của nó, rồi `steps`
bước đánh giá. Ước một epoch = số bước train × giây / bước + 2 lượt validation (val loss và AUC
validation mỗi epoch, như train.py). Tổng tối đa = epoch × số epoch tối đa × số seed (dừng sớm sẽ
ngắn hơn). Kết quả: `runs/<RUN>/benchmark.json`.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch

from vn_av_df import tracking
from vn_av_df.common.runtime import environment, write_json
from vn_av_df.training.models import build
from vn_av_df.training.train import feature_dim, loaders, recipe

# facebook/wav2vec2-xls-r-300m (dùng khi chưa tải trọng số; chỉ để đo thời gian).
XLSR_300M = dict(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    feat_extract_norm="layer",
    do_stable_layer_norm=True,
    conv_bias=True,
    layerdrop=0.1,
)
# Số mẫu mỗi part theo thiết kế (docs/v003_generation.md §0): 480 clip real train × 9 file.
DESIGN = {"train": 4320, "validation": 1080, "train_real": 480, "validation_real": 120}
FRAMES = 175  # clip dài nhất 7 s


def synthetic_batch(model, size, frames, dim, device, speakers=24):
    """Batch ngẫu nhiên đúng kích thước đầu vào của model."""
    needs = model.needs
    rand = torch.rand
    batch = {
        "mask": torch.ones(size, frames, dtype=torch.bool),
        "index": torch.arange(size),
        "speaker": torch.randint(0, speakers, (size,)),
        "video": {
            t: torch.randint(0, 2, (size,)).float() for t in ("ai", "visual", "audio", "mismatch")
        },
    }
    for task in ("ai", "visual", "audio", "mismatch"):
        target = torch.zeros(size, frames)
        target[: size // 2, frames // 3 : frames // 2] = 1
        batch[f"y_{task}"] = target
    if "av" in needs:
        batch["a"], batch["v"] = torch.randn(size, frames, dim), torch.randn(size, frames, dim)
    if "ssl" in needs:
        batch["ssl"] = torch.randn(size, frames, 3, 1024)
    if needs & {"mouth", "pose"}:
        batch["lips"], batch["pose"] = rand(size, frames, 20, 2) * 0.1, rand(size, frames, 3) * 0.1
    if "mouth" in needs:
        batch["mouth"] = torch.randint(0, 255, (size, frames, 96, 96), dtype=torch.uint8)
    if "wave" in needs:
        batch["wave"] = torch.randn(size, frames, 640) * 0.1
    if "lipinc" in needs:
        batch["lipinc"] = torch.randint(0, 255, (size, 8, 64, 144, 3), dtype=torch.uint8)
    if model.references:
        refs = {
            k: v[:, None].repeat(1, model.references, *([1] * (v.dim() - 1)))[:, :, :50]
            for k, v in batch.items()
            if k in ("a", "v", "ssl", "mouth", "lips", "pose")
        }
        refs["valid"] = torch.ones(size, model.references, 50, dtype=torch.bool)
        batch["refs"], batch["ref_count"] = refs, torch.full((size,), model.references)
    return batch


def to(batch, device):
    if isinstance(batch, dict):
        return {k: to(v, device) for k, v in batch.items()}
    return batch.to(device, non_blocking=True) if torch.is_tensor(batch) else batch


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def measure(model, batches, steps, warmup, device, optimizer=None, scaler=None, clip=None):
    """Giây / bước (tính toán) và giây / bước (chờ dữ liệu), bỏ `warmup` bước đầu."""
    compute, waiting = [], []
    model.train(optimizer is not None)
    amp = scaler is not None
    iterator = iter(batches)
    for step in range(warmup + steps):
        tick = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch = to(batch, device)
        sync(device)
        ready = time.perf_counter()
        with torch.set_grad_enabled(optimizer is not None), torch.autocast("cuda", enabled=amp):
            out = model(batch)
            loss = model.loss(batch, out, None) if optimizer is not None else None
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            (scaler.scale(loss) if amp else loss).backward()
            if amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        sync(device)
        if step >= warmup:
            waiting.append(ready - tick)
            compute.append(time.perf_counter() - ready)
    if not compute:
        return None, None
    return float(np.mean(compute)), float(np.mean(waiting))


def benchmark(cfg, models=None, steps=10, warmup=2, synthetic=False):
    models = list(models or cfg["models"])
    device = cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"
    rows = None
    if not synthetic:
        from vn_av_df.dataset import training_dataset
        from vn_av_df.training.data import speaker_index

        rows, _ = training_dataset(cfg)
        speakers = speaker_index(rows)
    assets = dict(cfg.get("model_assets") or {})
    if synthetic and not (assets.get("xlsr") and Path(str(assets["xlsr"])).exists()):
        assets["xlsr"] = XLSR_300M
    init = assets.get("avh_align_init")
    assets["load_pretrained"] = bool(init and Path(init).is_file()) or not synthetic
    base = cfg["training"]
    seeds = len(cfg["seeds"])
    dim = feature_dim(cfg, rows) if rows else 1024
    results = {}
    for name in models:
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        n_speakers = len(speakers) if rows else 24
        model = build(
            name, dim, base["hidden"], base["dropout"], {**assets, "n_speakers": n_speakers}
        )
        model = model.to(device)
        options = recipe(cfg, model)
        if not model.trainable:
            results[name] = {"trainable": False}
            continue
        if rows is not None:
            train_rows = [r for r in rows if r["split"] == "train"]
            val_rows = [r for r in rows if r["split"] == "validation"]
            if model.real_only:
                train_rows = [r for r in train_rows if r.get("cell") == "real"]
                val_rows = [r for r in val_rows if r.get("cell") == "real"]
            n_train, n_val = len(train_rows), len(val_rows)
            _, train_batches = loaders(cfg, model, train_rows, 0, True, options, speakers)
            _, val_batches = loaders(cfg, model, val_rows, 0, False, options, speakers)
            val_batches = torch.utils.data.DataLoader(
                val_batches.dataset,
                batch_size=base["eval_batch_size"],
                collate_fn=val_batches.collate_fn,
                num_workers=base.get("workers", 0),
            )
        else:
            n_train = DESIGN["train_real" if model.real_only else "train"]
            n_val = DESIGN["validation_real" if model.real_only else "validation"]
            frames = min(model.crop_frames or FRAMES, FRAMES)
            train_batch = synthetic_batch(model, options["batch_size"], frames, dim, device)
            val_batch = synthetic_batch(model, base["eval_batch_size"], FRAMES, dim, device)
            train_batches = [train_batch] * (warmup + steps)
            val_batches = [val_batch] * (warmup + steps)
        groups = (
            model.optimizer_groups(options["lr"])
            if hasattr(model, "optimizer_groups")
            else model.parameters()
        )
        kind = torch.optim.Adam if options.get("optimizer") == "adam" else torch.optim.AdamW
        optimizer = kind(groups, lr=options["lr"], eps=options.get("adam_eps", 1e-8))
        amp = options.get("amp") and str(device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda") if amp else None
        error = None
        try:
            train_s, train_wait = measure(
                model, train_batches, steps, warmup, device, optimizer, scaler
            )
            with torch.no_grad():
                eval_s, eval_wait = measure(model, val_batches, steps, warmup, device)
        except torch.cuda.OutOfMemoryError as exc:  # báo rõ model nào không vừa GPU
            error, train_s, train_wait, eval_s, eval_wait = (
                f"out of memory: {exc}",
                None,
                None,
                None,
                None,
            )
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        entry = {
            "trainable": True,
            "batch_size": options["batch_size"],
            "max_epochs": options["epochs"],
            "train_samples": n_train,
            "val_samples": n_val,
            "peak_gpu_gb": peak,
            "parameters": sum(p.numel() for p in model.parameters()),
            "error": error,
        }
        if train_s is not None and eval_s is not None:
            train_steps = math.ceil(n_train / options["batch_size"])
            eval_steps = math.ceil(n_val / base["eval_batch_size"])
            per_step_train = train_s + (train_wait or 0)
            per_step_eval = eval_s + (eval_wait or 0)
            epoch_s = train_steps * per_step_train + 2 * eval_steps * per_step_eval
            entry.update(
                {
                    "train_step_s": train_s,
                    "train_wait_s": train_wait,
                    "eval_step_s": eval_s,
                    "eval_wait_s": eval_wait,
                    "epoch_minutes": epoch_s / 60,
                    "max_hours_per_seed": epoch_s * options["epochs"] / 3600,
                    "max_hours_all_seeds": epoch_s * options["epochs"] * seeds / 3600,
                }
            )
        results[name] = entry
        print(
            f"{name:22s} batch {entry['batch_size']:>3} | epoch "
            f"{entry.get('epoch_minutes', float('nan')):6.1f} phút | tối đa "
            f"{entry.get('max_hours_per_seed', float('nan')):6.1f} giờ / seed | GPU "
            f"{(peak or 0):5.1f} GB" + (f" | {error[:60]}" if error else ""),
            flush=True,
        )
        del model, optimizer
    report = {
        "mode": "synthetic" if synthetic else "cache",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": seeds,
        "steps": steps,
        "models": results,
        "max_hours_total": sum(r.get("max_hours_all_seeds", 0) for r in results.values()),
    }
    report["environment"] = environment()
    write_json(Path(cfg["runs"]) / "benchmark.json", report)
    run = tracking.open_run(cfg, "benchmark", f"{Path(cfg['runs']).name}-benchmark")
    if run.active:
        columns = ["model", "batch_size", "epoch_minutes", "max_hours_per_seed", "peak_gpu_gb"]
        run.table(
            "benchmark/models",
            columns + ["error"],
            [
                [name] + [r.get(c) for c in columns[1:]] + [r.get("error")]
                for name, r in results.items()
            ],
        )
        run.summary({"benchmark/max_hours_total": report["max_hours_total"], "gpu": report["gpu"]})
        run.finish()
    return report
