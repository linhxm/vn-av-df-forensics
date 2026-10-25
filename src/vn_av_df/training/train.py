"""Train model v003 trên cache: mỗi (model, seed) một thư mục `runs/<RUN>/<model>_seed<seed>/`.

- Model không train (speechforensics) chỉ ghi checkpoint cấu hình.
- Công thức train = cfg["training"] ghi đè bởi `model.train_options` (baseline giữ đúng optimizer,
  lr, batch, epoch, scheduler của code gốc); cfg["training"]["max_epochs"] (nếu có) chặn trên số
  epoch (dùng cho test nhanh).
- Model `real_only` (avh_align) chỉ dùng clip real của train / validation.
- Chọn checkpoint theo validation loss; early stopping `patience` epoch.
- Train tiếp được giữa chừng: sau mỗi epoch ghi `last.pt` (model, optimizer, scheduler, AMP scaler,
  lịch sử, trạng thái ngẫu nhiên); chạy lại thì tiếp từ epoch sau đó, xong thì xoá `last.pt`.
- Nguồn gốc lần train ghi ở `runs/<RUN>/provenance.json` (dữ liệu, chữ ký cache, commit code, tài
  sản); pipeline train không ghi vào ledger của data_pipeline (ledger chỉ cho dữ liệu real).
- Đã có `best.pt` + `history.json` hoàn chỉnh thì bỏ qua (chạy lại sau khi ngắt sẽ tiếp tục
  các model chưa xong; model đang dở train lại từ đầu).
"""

import io
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from vn_av_df import tracking
from vn_av_df.common.runtime import atomic_bytes, environment, read_json, write_json
from vn_av_df.dataset import training_dataset
from vn_av_df.progress import Progress, duration, system_status
from vn_av_df.training.data import (
    BenchData,
    cache_file,
    cache_roots,
    collate,
    speaker_index,
    to_device,
)
from vn_av_df.training.models import build


def feature_dim(cfg, rows):
    for row in rows:
        path = cache_file(cfg, "avhubert", row["sample_id"])
        if path.is_file():
            with np.load(path, allow_pickle=False) as z:
                return int(z["audio"].shape[1])
    return 1024


def pos_weights(rows):
    out = {}
    for task, key in (
        ("ai", "label"),
        ("visual", "synthetic_visual"),
        ("audio", "synthetic_audio"),
    ):
        counts = Counter(bool(r.get(key)) for r in rows)
        out[task] = min(5.0, max(0.2, counts[False] / max(1, counts[True])))
    counts = Counter(bool(r.get("av_mismatch_intervals")) for r in rows)
    out["mismatch"] = min(5.0, max(0.2, counts[False] / max(1, counts[True])))
    return out


def speaker_weights(rows):
    """Trọng số lấy mẫu khi train = 1 / số clip gốc của người đó, để mỗi người đóng góp như nhau
    dù có 15-20 clip (docs/v003_model.md §7). None khi mọi người cùng số clip (giữ shuffle thường)."""
    clips = defaultdict(set)
    for r in rows:
        clips[(r["split"], r.get("speaker_id"))].add(r.get("source_clip_id") or r["sample_id"])
    sizes = {key: len(value) for key, value in clips.items()}
    if len(set(sizes.values())) <= 1:
        return None
    return [1.0 / sizes[(r["split"], r.get("speaker_id"))] for r in rows]


def save(path, state):
    stream = io.BytesIO()
    torch.save(state, stream)
    atomic_bytes(path, stream.getvalue())


def recipe(cfg, model):
    """Tuỳ chọn train của model: mặc định chung, ghi đè bởi công thức gốc của từng paper."""
    options = {**cfg["training"], **model.train_options}
    cap = cfg["training"].get("max_epochs")
    if cap:
        options["epochs"] = min(options["epochs"], cap)
    return options


def loaders(cfg, model, rows, seed, train, options=None, speakers=None):
    options = options or recipe(cfg, model)
    data = BenchData(
        cfg,
        rows,
        model.needs,
        augment=train and model.augment,
        lag_augment=train and getattr(model, "lag_augment", False),
        references=model.references,
        fixed_references=not train,
        seed=seed,
        crop_frames=model.crop_frames if train else None,
        pseudo_fake=model.pseudo_fake if train else 0.0,
        rawboost=train and model.rawboost,
        speakers=speakers,
    )
    generator = torch.Generator().manual_seed(seed)
    weights = speaker_weights(data.rows) if train and options.get("speaker_balance") else None
    sampler = (
        torch.utils.data.WeightedRandomSampler(weights, len(weights), generator=generator)
        if weights
        else None
    )
    return data, torch.utils.data.DataLoader(
        data,
        batch_size=options["batch_size"],
        shuffle=train and sampler is None,
        sampler=sampler,
        collate_fn=collate,
        num_workers=options.get("workers", 0),
        generator=generator,
    )


def run_epoch(
    model, loader, optimizer, device, weights, stats=None, clip=5.0, scaler=None, label=None
):
    """Loss trung bình; `stats` (dict) nhận thêm trung bình từng thành phần loss + grad norm.
    `label`: in tiến độ theo batch (batch đầu, rồi mỗi 60 s) kèm loss trung bình tới lúc đó."""
    total, count = 0.0, 0
    parts, norms, skipped = Counter(), [], 0
    model.train(optimizer is not None)
    amp = scaler is not None and str(device).startswith("cuda")
    progress = (
        Progress(label, len(loader), unit="batch", every_n=10**9, heartbeat=False)
        if label
        else None
    )
    for batch in loader:
        batch = to_device(batch, device)
        with torch.set_grad_enabled(optimizer is not None), torch.autocast("cuda", enabled=amp):
            out = model(batch)
            loss = model.loss(batch, out, weights)
        if optimizer is not None and loss.requires_grad:
            optimizer.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            params = [p for p in model.parameters() if p.grad is not None]
            norm = float(torch.nn.utils.clip_grad_norm_(params, clip if clip else float("inf")))
            # AMP: gradient tràn số (inf / NaN) ở vài bước đầu khi GradScaler dò hệ số; scaler bỏ
            # qua bước đó. Chỉ lấy trung bình norm hữu hạn, đếm riêng số bước tràn (BL-A, 25/10).
            if math.isfinite(norm):
                norms.append(norm)
            else:
                skipped += 1
            if amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        size = len(batch["index"])
        total += float(loss.detach()) * size
        count += size
        for key, value in (getattr(model, "loss_parts", None) or {}).items():
            parts[key] += value * size
        if progress:
            progress.step(note=f"loss {total / max(1, count):.4f}")
    if stats is not None:
        stats.update({k: v / max(1, count) for k, v in parts.items()})
        if norms:
            stats["grad_norm"] = float(np.mean(norms))
        if skipped:
            stats["grad_nonfinite_steps"] = skipped
    return total / max(1, count)


def validation_metrics(model, cfg, rows, device):
    """AUC validation theo tác vụ và theo ô / generator / giọng (tác vụ ai), mỗi epoch."""
    from vn_av_df.training.evaluate import TASKS, auc, predict, target, task_scores

    predictions = predict(model, cfg, rows, device)
    out = {}
    scores_by_task = {}
    for task in TASKS:
        scores = {s: task_scores(model, p).get(task) for s, p in predictions.items()}
        if any(v is None for v in scores.values()):
            continue
        scores_by_task[task] = scores
        out[f"val_auc/{task}"] = auc(
            [target(r, task) for r in rows], [scores[r["sample_id"]] for r in rows]
        )
    ai = scores_by_task.get("ai")
    if ai:
        reals = [r for r in rows if r.get("cell") == "real"]
        for key in ("cell", "generator", "voice_method", "kind"):
            groups = {}
            for r in rows:
                if r.get("label") and r.get(key) not in (None, "none"):
                    groups.setdefault(r[key], []).append(r)
            for value, members in groups.items():
                labels = [1] * len(members) + [0] * len(reals)
                values = [ai[r["sample_id"]] for r in members + reals]
                out[f"val_auc_ai_by_{key}/{value}"] = auc(labels, values)
    model.train()
    return out


def train_one(cfg, name, seed, rows, out):
    out = Path(out)
    if (out / "best.pt").is_file() and (out / "history.json").is_file():
        if read_json(out / "history.json").get("complete"):
            return read_json(out / "history.json")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"
    dim = feature_dim(cfg, rows)
    speakers = speaker_index(rows)
    assets = {**(cfg.get("model_assets") or {}), "n_speakers": len(speakers)}
    base = cfg["training"]
    model = build(name, dim, base["hidden"], base["dropout"], assets=assets).to(device)
    options = recipe(cfg, model)
    config = {
        "name": name,
        "dim": dim,
        "hidden": base["hidden"],
        "dropout": base["dropout"],
        "assets": model.rebuild_assets(),
    }
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    if model.real_only:  # avh_align: chỉ học từ real (đúng paper)
        train_rows = [r for r in train_rows if r["label"] == 0 and r.get("cell") == "real"]
        val_rows = [r for r in val_rows if r["label"] == 0 and r.get("cell") == "real"]
    history = {
        "model": name,
        "seed": seed,
        "epochs": [],
        "complete": False,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
    }
    run = tracking.open_run(
        cfg,
        "train",
        f"{name}_seed{seed}",
        group=cfg.get("run_group") or Path(cfg["runs"]).name,
        config={
            "model": name,
            "seed": seed,
            **config,
            "training": options,
            "parameters": sum(p.numel() for p in model.parameters()),
            "real_only": model.real_only,
            "pseudo_fake": model.pseudo_fake,
            "needs": sorted(model.needs),
            "augment_mouth": model.augment,
            "augment_lag": getattr(model, "lag_augment", False),
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
        },
        folder=out,
        tags=[name],
    )
    if not model.trainable:
        save(out / "best.pt", {"config": config, "state": model.state_dict(), "epoch": 0})
        history["complete"] = True
        history["validation"] = validation_metrics(model, cfg, val_rows, device)
        write_json(out / "history.json", history)
        run.log(history["validation"])
        run.summary({"trainable": False})
        run.finish()
        return history
    weights = {k: torch.tensor(v, device=device) for k, v in pos_weights(train_rows).items()}
    _, train_loader = loaders(cfg, model, train_rows, seed, True, options, speakers)
    _, val_loader = loaders(cfg, model, val_rows, seed, False, options, speakers)
    lr = options["lr"]
    groups = (
        model.optimizer_groups(lr) if hasattr(model, "optimizer_groups") else model.parameters()
    )
    kind = torch.optim.Adam if options.get("optimizer") == "adam" else torch.optim.AdamW
    optimizer = kind(
        groups, lr=lr, weight_decay=options["weight_decay"], eps=options.get("adam_eps", 1e-8)
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=options.get("plateau_patience", 5)
        )
        if options.get("scheduler") == "plateau"
        else None
    )
    amp = options.get("amp") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if amp else None
    clip = options.get("clip", 5.0)
    best, patience, stop, start = None, 0, "max_epochs", 1
    resume = out / "last.pt"
    if resume.is_file():
        state = torch.load(resume, map_location=device, weights_only=False)
        speed = ("workers", "eval_batch_size")  # chỉ ảnh hưởng tốc độ, đổi được khi chạy tiếp
        same = {k: v for k, v in options.items() if k not in speed} == {
            k: v for k, v in state["options"].items() if k not in speed
        }
        if state["config"] != config or not same:
            raise ValueError(
                f"{resume}: cấu hình model / công thức train khác lần train dở; "
                f"chuyển thư mục {out} đi chỗ khác để train lại từ đầu"
            )
        model.load_state_dict(state["state"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        train_loader.generator.set_state(state["rng"]["loader"])
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        history, best, patience = state["history"], state["best"], state["patience"]
        start = state["epoch"] + 1
        history.setdefault("resumed_at", []).append(start)
        print(f"{name} seed{seed}: train tiếp từ epoch {start}", flush=True)
    print(
        f"{name} seed{seed}: {sum(p.numel() for p in model.parameters()):,} tham số | "
        f"{len(train_rows)} train / {len(val_rows)} val, {len(train_loader)} batch / epoch | "
        f"{options.get('optimizer', 'adamw')} lr {lr:g}, batch {options['batch_size']}, tối đa "
        f"{options['epochs']} epoch, dừng sớm {options['patience']} | AMP {bool(amp)} | "
        f"real_only {model.real_only}, pseudo-fake {model.pseudo_fake} | {device} | {system_status()}",
        flush=True,
    )
    began = time.monotonic()
    for epoch in range(start, options["epochs"] + 1):
        tick = time.monotonic()
        train_loader.dataset.epoch = epoch
        model.on_epoch(epoch, options["epochs"])
        train_stats, val_stats = {}, {}
        tag = f"{name} seed{seed} epoch {epoch}/{options['epochs']}"
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            weights,
            train_stats,
            clip,
            scaler,
            label=f"{tag} train",
        )
        val_loss = run_epoch(
            model, val_loader, None, device, weights, val_stats, label=f"{tag} val"
        )
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise RuntimeError(
                f"{tag}: loss không hữu hạn (train {train_loss}, val {val_loss}); last.pt giữ "
                "epoch trước"
            )
        if scheduler is not None:
            scheduler.step(val_loss)
        metrics = validation_metrics(model, cfg, val_rows, device)
        if torch.cuda.is_available():
            # Trả bộ nhớ đệm của lượt chấm (cửa sổ dài, không graph) trước epoch train sau.
            torch.cuda.empty_cache()
        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "seconds": round(time.monotonic() - tick, 1),
                "train_parts": train_stats,
                "val_parts": val_stats,
                "validation": metrics,
            }
        )
        run.log(
            {
                "epoch": epoch,
                "loss/train": train_loss,
                "loss/val": val_loss,
                **{f"loss_train/{k}": v for k, v in train_stats.items() if k != "grad_norm"},
                **{f"loss_val/{k}": v for k, v in val_stats.items() if k != "grad_norm"},
                "grad_norm": train_stats.get("grad_norm"),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.monotonic() - tick,
                **metrics,
                **(
                    {"gpu/max_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
                    if torch.cuda.is_available()
                    else {}
                ),
            },
            step=epoch,
        )
        improved = best is None or val_loss < best - 1e-4
        spent = time.monotonic() - began
        per_epoch = spent / (epoch - start + 1)
        aucs = " ".join(
            f"{k.split('/')[1]} {v:.3f}"
            for k, v in metrics.items()
            if k.startswith("val_auc/") and v is not None
        )
        status = "tốt nhất" if improved else f"chưa tốt hơn {patience + 1}/{options['patience']}"
        print(
            f"{tag}: train {train_loss:.4f} val {val_loss:.4f} ({status})"
            f" | AUC val {aucs or '-'} | lr {optimizer.param_groups[0]['lr']:.2g}"
            f" | grad {train_stats.get('grad_norm', float('nan')):.2f}"
            f" | {duration(time.monotonic() - tick)} / epoch, còn tối đa "
            f"~{duration(per_epoch * (options['epochs'] - epoch))} | {system_status()}",
            flush=True,
        )
        if best is None or val_loss < best - 1e-4:
            best, patience = val_loss, 0
            save(
                out / "best.pt",
                {
                    "config": config,
                    "state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
            )
        else:
            patience += 1
            if patience >= options["patience"]:
                stop = f"early_stop (patience {options['patience']})"
                break
        write_json(out / "history.json", history)
        save(
            resume,
            {
                "config": config,
                "options": options,
                "epoch": epoch,
                "state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "history": history,
                "best": best,
                "patience": patience,
                "rng": {
                    "loader": train_loader.generator.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                },
            },
        )
        # Dừng sạch trước giới hạn phiên (Kaggle cắt ở 12 giờ, có thể mất Output): epoch sau không
        # kịp thì dừng sau khi đã lưu last.pt; phiên sau (PREVIOUS_RUN) train tiếp từ epoch sau.
        deadline = cfg.get("deadline")
        if deadline and time.time() + per_epoch > deadline:
            stop = "time_limit"
            break
    if stop == "time_limit":
        history["stop"] = stop
        write_json(out / "history.json", history)
        print(
            f"{name} seed{seed} TẠM DỪNG (hết MAX_HOURS) sau epoch {epoch}: last.pt đã lưu, phiên "
            f"sau đặt PREVIOUS_RUN = Output phiên này để train tiếp | {system_status()}",
            flush=True,
        )
        run.summary({"stop": stop, "epochs_run": len(history["epochs"])})
        run.finish()
        return history
    history["complete"] = True
    history["best_val_loss"] = best
    history["stop"] = stop
    write_json(out / "history.json", history)
    resume.unlink(missing_ok=True)  # checkpoint tạm (BL-A ~3,6 GB vì có trạng thái Adam của XLS-R)
    best_epoch = torch.load(out / "best.pt", map_location="cpu", weights_only=False)["epoch"]
    chosen = next(e for e in history["epochs"] if e["epoch"] == best_epoch)
    print(
        f"{name} seed{seed} XONG: epoch tốt nhất {best_epoch} (val {best:.4f}), dừng: {stop}, "
        f"{len(history['epochs'])} epoch trong {duration(time.monotonic() - began)} | "
        + " ".join(
            f"{k.split('/')[1]} {v:.3f}"
            for k, v in chosen["validation"].items()
            if k.startswith("val_auc/") and v is not None
        ),
        flush=True,
    )
    run.summary(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best,
            "stop": stop,
            "epochs_run": len(history["epochs"]),
            "train_hours": (time.monotonic() - began) / 3600,
            **{f"best/{k}": v for k, v in chosen["validation"].items()},
        }
    )
    run.artifact(
        f"model-{cfg.get('run_group') or Path(cfg['runs']).name}-{name}-seed{seed}",
        "model",
        [out / "best.pt", out / "history.json"],
        {"model": name, "seed": seed, "best_epoch": best_epoch, "best_val_loss": best},
    )
    run.finish()
    return history


def load_model(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    config = state["config"]
    assets = {**config.get("assets", {}), "load_pretrained": False}
    model = build(config["name"], config["dim"], config["hidden"], config["dropout"], assets)
    model.load_state_dict(state["state"])
    return model.to(device).eval()


def git_state(root):
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(cfg, rows, selection):
    """Nguồn gốc của lần train (thay cho ledger): dữ liệu, chữ ký từng cache, code, tài sản."""
    from concurrent.futures import ThreadPoolExecutor

    from vn_av_df.common.runtime import sha
    from vn_av_df.training.prepare import encoders_for

    def signature(meta):
        return read_json(meta).get("feature_signature")

    ids = {r["sample_id"] for r in rows}
    caches = {}
    for encoder in encoders_for(cfg["models"]):
        tick = time.monotonic()
        # Mọi gốc cache có encoder này (mỗi part một dataset cache): đếm mẫu và chữ ký trên tất cả.
        folders = [r / encoder for r in cache_roots(cfg) if (r / encoder).is_dir()]
        metas = {}
        for folder in folders:
            for meta in folder.glob("*.json"):
                if meta.stem in ids and meta.stem not in metas:
                    metas[meta.stem] = meta
        # Đọc song song: /kaggle/input đọc từng file nhỏ có độ trễ; T-B (4 encoder × 7.622 file)
        # đứng >17 phút không in gì ở bước này (25/10).
        with ThreadPoolExecutor(max_workers=16) as pool:
            signatures = Counter(pool.map(signature, metas.values()))
        print(
            f"provenance: {encoder} {len(metas)}/{len(ids)} mẫu, {len(signatures)} chữ ký"
            f" | {duration(time.monotonic() - tick)}",
            flush=True,
        )
        caches[encoder] = {
            "folders": [str(f) for f in folders],
            "samples_found": len(metas),
            "samples_needed": len(ids),
            "feature_signatures": dict(signatures),
            "extraction": [
                read_json(f / "extraction.json")
                for f in folders
                if (f / "extraction.json").is_file()
            ],
        }
    here = Path(__file__).resolve()
    code = {p.name: sha(p) for p in sorted(here.parent.glob("*.py"))}
    code["encoders.py"] = sha(here.parents[1] / "features/encoders.py")
    assets = {}
    for key, value in (cfg.get("model_assets") or {}).items():
        if isinstance(value, str) and Path(value).exists():
            path = Path(value)
            record = path / "vn_av_revision.json" if path.is_dir() else None
            receipt = path.with_suffix(path.suffix + ".download.json")
            assets[key] = (
                read_json(record)
                if record and record.is_file()
                else read_json(receipt)
                if receipt.is_file()
                else {"path": value}
            )
    return {
        "selection": selection,
        "models": cfg["models"],
        "seeds": cfg["seeds"],
        "options": cfg["training"],
        "caches": caches,
        "git": git_state(here.parents[3]),
        "code_sha256": code,
        "assets": assets,
        "environment": environment(),
        # Không dò siêu tham số: baseline dùng công thức của code gốc (models.*.train_options),
        # M1 / M2 đặt trước (settings.training); checkpoint chọn theo val loss.
        "hyperparameter_search": None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_state(folder):
    """(xong?, số epoch đã có) của một thư mục <model>_seed<seed>."""
    history = Path(folder) / "history.json"
    if not history.is_file():
        return (False, 0)
    data = read_json(history)
    return (bool(data.get("complete")), len(data.get("epochs", [])))


def merge_runs(target, sources, move=False):
    """Gộp output train của nhiều notebook (chạy song song, mỗi notebook vài model) vào `target`.

    Mỗi <model>_seed<seed>: giữ bản xong, hoặc bản nhiều epoch hơn (kèm last.pt để train tiếp).
    provenance.json nối danh sách (không mất lần chạy nào). Không chép kết quả test
    (test-lock / evaluation / report / gates): test chỉ chạy một lần khi đủ mọi model.
    move=True: chuyển thay vì chép (lane cùng ổ đĩa; checkpoint BL-A vài GB không bị nhân đôi).
    """
    import shutil

    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    taken = {}  # <model>_seed<seed> → notebook giữ bản cuối
    records = (
        read_json(target / "provenance.json") if (target / "provenance.json").is_file() else []
    )
    for source in map(Path, sources):
        for folder in sorted(p for p in source.iterdir() if p.is_dir()):
            mine = target / folder.name
            if run_state(folder) > run_state(mine):
                shutil.rmtree(mine, ignore_errors=True)
                if move:
                    shutil.move(str(folder), str(mine))
                else:
                    shutil.copytree(folder, mine)
                taken[folder.name] = source.name
        if (source / "provenance.json").is_file():
            records += [r for r in read_json(source / "provenance.json") if r not in records]
        for name in ("benchmark.json", "training.json"):
            if (source / name).is_file() and not (target / name).exists():
                shutil.copy2(source / name, target / name)
    write_json(target / "provenance.json", records)
    return taken


def missing_runs(cfg):
    """<model>_seed<seed> chưa train xong trong cfg["runs"] (chạy test khi danh sách này rỗng)."""
    runs = Path(cfg["runs"])
    return [
        f"{name}_seed{seed}"
        for name in cfg["models"]
        for seed in cfg["seeds"]
        if not run_state(runs / f"{name}_seed{seed}")[0]
    ]


def train_all(cfg):
    tick = time.monotonic()
    rows, selection = training_dataset(cfg)
    print(f"train: đọc dataset {len(rows)} mẫu | {duration(time.monotonic() - tick)}", flush=True)
    runs = Path(cfg["runs"])
    write_json(
        runs / "training.json",
        {
            "selection": selection,
            "models": cfg["models"],
            "seeds": cfg["seeds"],
            "options": cfg["training"],
        },
    )
    record = provenance(cfg, rows, selection)
    previous = runs / "provenance.json"
    if previous.is_file():  # mỗi lần chạy (kể cả chạy tiếp) thêm một mục, không ghi đè
        history = read_json(previous)
        history = history if isinstance(history, list) else [history]
    else:
        history = []
    write_json(previous, [*history, record])
    report = {}
    jobs = [(name, seed) for name in cfg["models"] for seed in cfg["seeds"]]
    splits = Counter(r["split"] for r in rows)
    print(
        f"train: {len(jobs)} lượt ({len(cfg['models'])} model × seed {cfg['seeds']}) | dữ liệu "
        f"{dict(splits)} | {runs} | {record['environment'].get('gpus') or 'CPU'}",
        flush=True,
    )
    began = time.monotonic()
    deadline = cfg.get("deadline")
    for index, (name, seed) in enumerate(jobs, 1):
        folder = runs / f"{name}_seed{seed}"
        done = (folder / "history.json").is_file() and read_json(folder / "history.json").get(
            "complete"
        )
        if not done and deadline and time.time() > deadline:
            print(
                f"=== [{index}/{len(jobs)}] {name} seed{seed}: hết MAX_HOURS, để phiên sau",
                flush=True,
            )
            report[f"{name}_seed{seed}"] = {"stop": "time_limit", "epochs": run_state(folder)[1]}
            continue
        print(
            f"=== [{index}/{len(jobs)}] {name} seed{seed}"
            f"{': đã xong, bỏ qua' if done else ''} | đã chạy {duration(time.monotonic() - began)}",
            flush=True,
        )
        history = train_one(cfg, name, seed, rows, folder)
        report[f"{name}_seed{seed}"] = {
            "epochs": len(history["epochs"]),
            "best_val_loss": history.get("best_val_loss"),
            "stop": history.get("stop"),
        }
    print(f"train xong {len(jobs)} lượt trong {duration(time.monotonic() - began)}", flush=True)
    return report
