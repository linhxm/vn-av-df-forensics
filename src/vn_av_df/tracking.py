"""Tiến độ train theo từng epoch: luôn in ra console, ghi thêm lên W&B khi cfg["wandb"] được đặt.

Không đặt cfg["wandb"] thì chỉ in: local/test không cần cài wandb.
Mỗi detector/seed một run, mở suốt lúc train nên W&B ghi được cả log console và GPU/RAM theo thời gian.
ID run lưu ở <run>/<architecture>_seed<seed>/wandb.json để notebook bổ sung bảng/ảnh sau.
"""

import json
from datetime import datetime, timedelta, timezone

from vn_av_df.common.runtime import write_json

STAGES = ("stageA", "stageS", "detector")  # Tái dựng A→V, head sync, detector.
_run = None
_label = ""


def start(cfg, folder, architecture, seed, data=None):
    """Mở run cho một detector/seed; train tiếp sau khi bị ngắt thì mở run mới, ghi lại lịch sử cũ.

    data: số mẫu theo split/ô 2×2 đưa vào config để biết run train trên dữ liệu nào.
    """
    global _run, _label
    _label = folder.name
    options = cfg.get("wandb")
    if not options:
        return
    import wandb

    finish()
    # Ngày giờ Việt Nam (máy Kaggle chạy UTC) trong tên run: phân biệt lần train và lần resume.
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%m%d-%H%M")
    settings = {
        "architecture": architecture,
        "seed": seed,
        "dataset_parts": cfg.get("dataset_parts"),
        "held_out_generators": cfg.get("held_out_generators", []),
        **{k: cfg.get(k) for k in ("training", "reconstruction", "sync", "encoder")},
        "artifact_encoder": cfg.get("artifact_encoder"),
        "data": data,
    }
    _run = wandb.init(
        project=options["project"],
        entity=options.get("entity"),
        group=options.get("group"),
        name="-".join(filter(None, (options.get("group"), folder.name, stamp))),
        job_type="train",
        config=json.loads(json.dumps(settings, default=str)),
    )
    for stage in STAGES:
        _run.define_metric(f"{stage}/epoch")
        _run.define_metric(f"{stage}/*", step_metric=f"{stage}/epoch")
    write_json(
        folder / "wandb.json", {"id": _run.id, "project": _run.project, "entity": _run.entity}
    )


def _text(value):
    return f"{value:.4f}" if isinstance(value, float) else "-" if value is None else str(value)


def log(stage, row, echo=True):
    """Một dòng history (có khoá epoch) của stage; W&B bỏ giá trị None (vd. AUC chưa tính được).

    echo=False khi ghi lại lịch sử cũ lúc resume (đã in ở phiên trước).
    """
    if echo:
        # Console chỉ in chỉ số chính; khoá có "/" (theo ô 2×2, theo nhánh) xem trên W&B/history.
        shown = " | ".join(
            f"{k} {_text(v)}" for k, v in row.items() if k != "epoch" and "/" not in k
        )
        print(f"{_label} {stage} epoch {row.get('epoch')}: {shown}", flush=True)
    if _run is not None:
        _run.log({f"{stage}/{k}": v for k, v in row.items() if v is not None})


def finish(summary=None):
    global _run
    if _run is None:
        return
    if summary:
        _run.summary.update(json.loads(json.dumps(summary, default=str)))
    _run.finish()
    _run = None
