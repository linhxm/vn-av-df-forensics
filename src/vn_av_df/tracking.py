"""Log W&B: trích feature, train, test, báo cáo (kèm setup của phiên, setup_log.py) làm bằng chứng.

Bật bằng cfg["wandb"] (notebook prepare / train đặt; project để private):
    {"project": "vn-av-df-v005", "entity": None, "mode": "online"}
Không đặt cfg["wandb"] thì mọi hàm ở đây không làm gì: local và test không cần cài wandb.
Không đẩy ảnh / video người thật; chỉ số liệu, bảng, biểu đồ và file kết quả.

Mỗi run ghi config + hash dữ liệu/code để tái lập; id run lưu ở `<thư mục>/wandb-<tên>.json` để
phiên sau mở lại đúng run (resume) khi chạy tiếp sau khi bị ngắt.
"""

import math
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from vn_av_df.common.runtime import read_json, write_json

PROJECT = "vn-av-df-v005"


class Run:
    """Một run W&B; `active` False thì mọi lệnh bỏ qua."""

    def __init__(self, wandb=None, run=None):
        self.wandb, self.run = wandb, run

    @property
    def active(self):
        return self.run is not None

    def log(self, data, step=None):
        if self.active and data:
            clean = {k: v for k, v in data.items() if v is not None and not _nan(v)}
            self.run.log(clean, step=step) if step is not None else self.run.log(clean)

    def summary(self, data):
        if self.active:
            for k, v in data.items():
                if v is not None and not _nan(v):
                    self.run.summary[k] = v

    def table(self, key, columns, rows):
        if self.active and rows:
            self.run.log(
                {key: self.wandb.Table(columns=list(columns), data=[list(r) for r in rows])}
            )

    def tables(self, tables):
        """{key: (cột, dòng)} log một lần, commit=False: không tăng step (run log epoch theo step)."""
        if self.active:
            data = {
                key: self.wandb.Table(columns=list(columns), data=[list(r) for r in rows])
                for key, (columns, rows) in tables.items()
                if rows
            }
            if data:
                self.run.log(data, commit=False)

    def hist(self, key, values):
        values = [float(v) for v in values if v is not None and not _nan(v)]
        if self.active and values:
            self.run.log({key: self.wandb.Histogram(values)})

    def line(self, key, xs, ys, x="x", y="y", title=None):
        if self.active and len(xs):
            table = self.wandb.Table(
                columns=[x, y], data=[[float(a), float(b)] for a, b in zip(xs, ys)]
            )
            self.run.log({key: self.wandb.plot.line(table, x, y, title=title or key)})

    def bar(self, key, counts, label="group", value="count", title=None):
        if self.active and counts:
            table = self.wandb.Table(
                columns=[label, value], data=[[str(k), v] for k, v in counts.items()]
            )
            self.run.log({key: self.wandb.plot.bar(table, label, value, title=title or key)})

    def artifact(self, name, kind, paths, metadata=None):
        if not self.active:
            return
        art = self.wandb.Artifact(name, type=kind, metadata=metadata or {})
        paths = [Path(p) for p in paths]
        # Nhiều model cùng tên file (metrics.json): mọi file trùng tên đặt dưới thư mục model.
        # Thiếu điều này W&B báo "Cannot add the same path twice" (test part 1, 26/10).
        names = Counter(p.name for p in paths if p.is_file())
        added = False
        for path in paths:
            if path.is_file():
                key = path.name if names[path.name] == 1 else f"{path.parent.name}/{path.name}"
                art.add_file(str(path), name=key)
                added = True
            elif path.is_dir():
                art.add_dir(str(path), name=path.name)
                added = True
        if added:
            self.run.log_artifact(art)

    def finish(self):
        if self.active:
            self.run.finish()
            self.run = None


def _nan(value):
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def open_run(cfg, job_type, name, group=None, config=None, folder=None, tags=(), with_setup=False):
    """Mở run; `folder` có thì lưu id để phiên sau tiếp tục đúng run. with_setup: đính kèm setup
    của phiên (setup_log.attach), cho run chính của phiên (prepare, train, labels, report)."""
    options = cfg.get("wandb")
    if not options:
        return Run()
    import wandb

    record = Path(folder) / f"wandb-{name}.json" if folder else None
    previous = read_json(record) if record and record.exists() else {}
    # Tên hiển thị có ngày giờ bắt đầu (giờ Việt Nam); chạy tiếp sau khi ngắt giữ đúng tên và id cũ.
    started = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d-%H%M")
    display = previous.get("display") or f"{started} {name}"
    try:
        run = wandb.init(
            project=options.get("project", PROJECT),
            entity=options.get("entity"),
            mode=options.get("mode", "online"),
            dir=options.get("dir") or os.environ.get("WANDB_DIR"),
            job_type=job_type,
            group=group,
            name=display,
            id=previous.get("id"),
            resume="allow",
            config=config or {},
            tags=list(tags),
            reinit=True,
        )
    except Exception as exc:  # key / quyền / mạng: log W&B là bản sao, file local vẫn ghi đủ
        print(
            f"CẢNH BÁO W&B: không mở được run {name} ({exc}); chạy tiếp không log W&B", flush=True
        )
        return Run()
    if record:
        write_json(
            record,
            {"id": run.id, "name": name, "display": display, "job_type": job_type, "group": group},
        )
    opened = Run(wandb, run)
    if with_setup:
        from vn_av_df.setup_log import attach

        attach(opened, cfg)
    return opened


# ---------------------------------------------------------------- nội dung dùng chung
def lag_curve(a, v, valid, max_lag=15):
    """Cos trung bình giữa tiếng frame t và hình frame t − δ, δ ∈ [−max_lag, max_lag] (như
    SpeechForensics); chỉ số i ↔ tiếng trễ δ = i − max_lag frame. NaN khi < 10 frame chồng lên."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    n = len(a)
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        lo, hi = max(0, lag), min(n, n + lag)
        if hi - lo < 10:
            scores.append(np.nan)
            continue
        keep = valid[lo:hi] & valid[lo - lag : hi - lag]
        sims = (a[lo:hi] * v[lo - lag : hi - lag]).sum(1)
        scores.append(float(sims[keep].mean()) if keep.any() else np.nan)
    return np.asarray(scores)


def lag_profile(a, v, valid, max_lag=15):
    """Độ lệch tốt nhất giữa đặc trưng tiếng và hình; dương = tiếng trễ. Trả (lag, độ nhọn =
    max − median)."""
    scores = lag_curve(a, v, valid, max_lag)
    if np.all(np.isnan(scores)):
        return None, None
    best = int(np.nanargmax(scores))
    return best - max_lag, float(np.nanmax(scores) - np.nanmedian(scores))


def curves(run, key, series, x="fpr", y="tpr", title=None):
    """Nhiều đường trên một biểu đồ: series = {tên: (xs, ys)}."""
    if not run.active or not series:
        return
    data = [[name, float(a), float(b)] for name, (xs, ys) in series.items() for a, b in zip(xs, ys)]
    table = run.wandb.Table(columns=["series", x, y], data=data)
    run.run.log({key: run.wandb.plot.line(table, x, y, stroke="series", title=title or key)})
