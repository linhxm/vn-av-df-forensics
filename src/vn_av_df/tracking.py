"""Ghi tiến trình train lên W&B theo từng epoch khi cfg["wandb"] được đặt (notebook Kaggle).

Không đặt cfg["wandb"] thì mọi hàm không làm gì: local/test không cần cài wandb.
Mỗi detector/seed một run, mở suốt lúc train nên W&B ghi được cả GPU/RAM theo thời gian.
ID run lưu ở <run>/<architecture>_seed<seed>/wandb.json để notebook bổ sung bảng/ảnh sau.
"""

import json

from vn_av_df.common.runtime import write_json

STAGES = ("stageA", "stageS", "detector")  # Tái dựng A→V, head sync, detector.
_run = None


def start(cfg, folder, architecture, seed):
    """Mở run cho một detector/seed; train tiếp sau khi bị ngắt thì mở run mới, ghi lại lịch sử cũ."""
    global _run
    options = cfg.get("wandb")
    if not options:
        return
    import wandb

    finish()
    settings = {
        "architecture": architecture,
        "seed": seed,
        "dataset_parts": cfg.get("dataset_parts"),
        "held_out_generators": cfg.get("held_out_generators", []),
        **{k: cfg.get(k) for k in ("training", "reconstruction", "sync", "encoder")},
        "artifact_encoder": cfg.get("artifact_encoder"),
    }
    _run = wandb.init(
        project=options["project"],
        entity=options.get("entity"),
        group=options.get("group"),
        name=f"{options['group']}-{folder.name}" if options.get("group") else folder.name,
        job_type="train",
        config=json.loads(json.dumps(settings, default=str)),
    )
    for stage in STAGES:
        _run.define_metric(f"{stage}/epoch")
        _run.define_metric(f"{stage}/*", step_metric=f"{stage}/epoch")
    write_json(
        folder / "wandb.json", {"id": _run.id, "project": _run.project, "entity": _run.entity}
    )


def log(stage, row):
    """Một dòng history (có khoá epoch) của stage; bỏ giá trị None (vd. AUC chưa tính được)."""
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
