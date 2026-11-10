"""Module `wandb` giả cho test: ghi lại mọi run, key đã log, video, artifact (không cần mạng)."""

import itertools


class Table:
    def __init__(self, columns, data):
        self.columns, self.data = columns, data


class Histogram:
    def __init__(self, values):
        self.values = list(values)


class Video:
    def __init__(self, path, caption=None, format=None):
        self.path, self.caption = path, caption


class Plot:
    def line(self, table, x, y, stroke=None, title=None):
        return ("line", table, x, y, stroke)

    def bar(self, table, label, value, title=None):
        return ("bar", table, label, value)


class Artifact:
    def __init__(self, name, type, metadata=None):
        self.name, self.type, self.metadata, self.files = name, type, metadata, []
        self.names = set()

    def add_file(self, path, name=None):
        # Như wandb thật: hai file cùng tên trong artifact thì lỗi (test part 1, 26/10).
        name = name or path.replace("\\", "/").rsplit("/", 1)[-1]
        if name in self.names:
            raise ValueError(f"Cannot add the same path twice: {name!r}")
        self.names.add(name)
        self.files.append(path)

    def add_dir(self, path, name=None):
        self.files.append(path)


class FakeConfig(dict):
    def update(self, data, allow_val_change=False):
        changed = [k for k, v in data.items() if k in self and self[k] != v]
        if changed and not allow_val_change:
            raise ValueError(f"Đổi giá trị config cần allow_val_change: {changed}")
        super().update(data)


class FakeRun:
    ids = itertools.count(1)

    def __init__(self, store, **kwargs):
        self.store, self.kwargs = store, kwargs
        self.id = kwargs.get("id") or f"run{next(self.ids)}"
        self.logged, self.summary, self.artifacts, self.finished = [], {}, [], False
        self.config, self.step = FakeConfig(kwargs.get("config") or {}), 0
        store.runs.append(self)

    def log(self, data, step=None, commit=None):
        # Như W&B thật: log theo step nhỏ hơn step hiện tại thì dữ liệu bị bỏ; test báo lỗi luôn.
        if step is not None:
            if step < self.step:
                raise ValueError(f"step {step} < step hiện tại {self.step}: W&B sẽ bỏ dữ liệu")
            self.step = step
        elif commit is not False:
            self.step += 1
        self.logged.append(dict(data))

    def define_metric(self, name, step_metric=None, **kwargs):
        self.axes = {**getattr(self, "axes", {}), name: step_metric}

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self):
        self.finished = True

    def keys(self):
        return {k for entry in self.logged for k in entry}


class FakeWandb:
    Table, Histogram, Video, Artifact = Table, Histogram, Video, Artifact

    def __init__(self):
        self.runs = []
        self.plot = Plot()

    def init(self, **kwargs):
        return FakeRun(self, **kwargs)

    def by_type(self, job_type):
        return [r for r in self.runs if r.kwargs.get("job_type") == job_type]
