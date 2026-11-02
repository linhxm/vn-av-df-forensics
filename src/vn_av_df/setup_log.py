"""Ghi cách cài một phiên (notebook Kaggle prepare / train / report) vào Output và vào run W&B.

v004 train tách từng model theo phiên, nên môi trường có thể khác nhau giữa các model. `log_setup`
ghi `<folder>/<phiên>/` (folder = runs/<RUN>/setup ở train, cache/.../setup ở prepare): setup.json
(phiên, commit, phần cứng, thư viện, biên nhận cài worker, dữ liệu / cache / tài sản gắn vào),
pip-freeze của kernel và từng worker, log cài (pip của notebook, setup.log của worker).

Không có run W&B riêng: chính run của phiên (`prepare`, `train`, `labels`, `report`) đính kèm qua
`tracking.open_run(..., with_setup=True)`: config `setup`, bảng `setup/*`, artifact `setup-log`.
Nhiều run cùng phiên (hai model) dùng chung một artifact, W&B không lưu trùng nội dung. Run train
tiếp ở phiên sau đính kèm thêm setup của phiên đó. `merge_runs` giữ log setup của mọi phiên.

Không ghi secret: chỉ tên biến môi trường W&B / Kaggle, không giá trị.
"""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from vn_av_df.common.runtime import environment, read_json, write_json

MAX_LOG_BYTES = 2_000_000  # mỗi log giữ phần cuối 2 MB (pip -v của worker có thể vài chục MB)
SECRET = re.compile(r"(api[_-]?key|token|secret|password)", re.I)


def pip_freeze(python=None, timeout=120):
    """Danh sách `tên==phiên bản` của một trình Python (mặc định kernel đang chạy)."""
    try:
        out = subprocess.run(
            [str(python or sys.executable), "-m", "pip", "freeze", "--all"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _tail(path, target):
    data = Path(path).read_bytes()
    target.write_bytes(data[-MAX_LOG_BYTES:])
    return {"file": target.name, "bytes": len(data), "kept": min(len(data), MAX_LOG_BYTES)}


def _clean(value):
    """Bỏ khoá trông như secret (không bao giờ ghi giá trị của chúng)."""
    if isinstance(value, dict):
        return {k: "***" if SECRET.search(str(k)) else _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def worker_receipts(env_root):
    """{tên worker: biên nhận cài} từ <env_root>/.kaggle-tools/<tên>/environment.json."""
    out = {}
    tools = Path(env_root) / ".kaggle-tools" if env_root else None
    if not tools or not tools.is_dir():
        return out
    for receipt in sorted(tools.glob("*/environment.json")):
        data = read_json(receipt)
        recipe = data.get("recipe")
        if isinstance(recipe, dict):  # bỏ nguyên văn script (dài), giữ công thức
            data["recipe"] = {k: v for k, v in recipe.items() if k != "script"}
        out[receipt.parent.name] = data
    return out


def worker_logs(env_root):
    tools = Path(env_root) / ".kaggle-tools" if env_root else None
    return sorted(tools.glob("*/setup.log")) if tools and tools.is_dir() else []


def log_setup(cfg, session, folder, logs=(), env_root=None, info=None, timings=None):
    """Ghi phiên `session` (vd. "train-P1-…", "prepare-p2-gpu"); trả {"session", "file"}: đặt vào
    cfg["setup"] để run W&B của phiên đính kèm.

    logs: file log của notebook (pip install, setup_avhubert…); env_root: thư mục môi trường worker
    (log + biên nhận + pip freeze của từng worker); info: thông tin riêng phiên (lane, dữ liệu,
    tài sản đã tải); timings: {bước: giây} của các cell cài đặt."""
    from vn_av_df.training.train import git_state

    root = Path(cfg.get("root") or Path(__file__).resolve().parents[2])
    out = Path(folder) / session
    out.mkdir(parents=True, exist_ok=True)
    kept = []
    names = set()
    for path in [*map(Path, logs), *worker_logs(env_root)]:
        if not path.is_file():
            continue
        name = path.name if path.name != "setup.log" else f"worker-{path.parent.name}-setup.log"
        if name in names:
            name = f"{path.parent.name}-{name}"
        names.add(name)
        kept.append({"source": str(path), **_tail(path, out / name)})
    kernel = pip_freeze()
    (out / "pip-freeze.txt").write_text("\n".join(kernel) + "\n", encoding="utf-8")
    receipts = worker_receipts(env_root)
    workers = {}
    for name, receipt in receipts.items():
        python = receipt.get("python")
        packages = pip_freeze(python) if python and Path(python).exists() else []
        if packages:
            (out / f"pip-freeze-{name}.txt").write_text("\n".join(packages) + "\n", "utf-8")
        workers[name] = {"receipt": receipt, "packages": len(packages)}
    group = cfg.get("run_group") or Path(cfg["runs"]).name
    record = _clean(
        {
            "session": session,
            "train_group": cfg.get("train_group"),
            "run_group": group,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git": git_state(root),
            "environment": environment(),
            "workers": workers,
            "dataset": {"root": cfg.get("dataset"), "parts": cfg.get("dataset_parts")},
            "cache": cfg.get("cache"),
            "runs": cfg.get("runs"),
            "device": cfg.get("device"),
            "training": cfg.get("training"),
            "mismatch_labels": cfg.get("mismatch_labels"),
            "timings_s": timings or {},
            "logs": kept,
            "info": info or {},
        }
    )
    write_json(out / "setup.json", record)
    result = {"session": session, "file": str(out / "setup.json")}
    print("setup log:", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def setup_record(cfg):
    """setup.json của phiên hiện tại (cfg["setup"]), None nếu chưa ghi."""
    path = Path((cfg.get("setup") or {}).get("file") or "")
    return read_json(path) if path.is_file() else None


def attach(run, cfg):
    """Đính kèm setup của phiên vào run W&B đang mở. Cập nhật config sau khi mở (cho phép đổi giá
    trị: run train tiếp ở phiên khác) và log bảng commit=False: không tăng step, các epoch log
    theo step=epoch không bị W&B bỏ vì step lùi."""
    record = setup_record(cfg)
    if not run.active or record is None:
        return
    folder = Path(cfg["setup"]["file"]).parent
    run.run.config.update(
        {
            "setup": {k: v for k, v in record.items() if k != "logs"},
            "session": record["session"],
            "train_group": record.get("train_group"),
        },
        allow_val_change=True,
    )
    freeze = folder / "pip-freeze.txt"
    kernel = freeze.read_text(encoding="utf-8").splitlines() if freeze.is_file() else []
    workers = record.get("workers") or {}
    kept = record.get("logs") or []
    run.tables(
        {
            "setup/kernel_packages": (
                ["package", "version"],
                [line.split("==", 1) if "==" in line else [line, ""] for line in kernel if line],
            ),
            "setup/workers": (
                ["worker", "status", "python", "packages"],
                [
                    [n, w["receipt"].get("status"), w["receipt"].get("python"), w["packages"]]
                    for n, w in workers.items()
                ],
            ),
            "setup/logs": (
                ["file", "bytes", "kept", "source"],
                [[k["file"], k["bytes"], k["kept"], k["source"]] for k in kept],
            ),
        }
    )
    group = record.get("run_group") or ""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", f"setup-{group}-{record['session']}")
    run.artifact(
        safe,
        "setup-log",
        sorted(p for p in folder.iterdir() if p.is_file()),
        {"session": record["session"]},
    )


def copy_tree(source, target):
    """Chép log setup của phiên trước (cache prepare) mà chưa có ở target."""
    source, target = Path(source), Path(target)
    if not source.is_dir():
        return []
    copied = []
    for session in sorted(p for p in source.iterdir() if p.is_dir()):
        if not (target / session.name).exists():
            shutil.copytree(session, target / session.name)
            copied.append(session.name)
    return copied
