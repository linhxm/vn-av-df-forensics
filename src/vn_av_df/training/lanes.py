"""Train song song nhiều nhóm model trong một phiên (Kaggle T4 ×2). v005: số lane có thể lớn hơn số GPU:
lane đầu dùng riêng GPU0 (model nặng), các lane sau chia GPU còn lại (`lane_devices`), vd. baseline nhẹ
chạy đồng thời trên GPU1.

Mỗi lane là một tiến trình `train_all` riêng (CUDA_VISIBLE_DEVICES = GPU của lane), ghi vào
`<runs>-gpu<i>`; xong thì chuyển (không chép) các <model>_seed<seed> về `cfg["runs"]` bằng
`merge_runs(move=True)`. Không đổi phép toán của model: chỉ chạy hai model cùng lúc thay vì lần
lượt, nên giờ phiên (quota Kaggle tính theo giờ phiên) giảm gần một nửa khi hai lane cân nhau.
"""

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from vn_av_df.common.runtime import read_json, write_json


def lane_dirs(cfg, count):
    runs = Path(cfg["runs"])
    return [runs.parent / f"{runs.name}-gpu{i}" for i in range(count)]


def lane_devices(lanes, gpus):
    """GPU của từng lane: lane 0 → GPU0 riêng; lane sau xoay vòng trên GPU 1..n-1 (chỉ 1 GPU: mọi lane GPU0)."""
    if gpus <= 1:
        return [0] * lanes
    return [0] + [1 + (i - 1) % (gpus - 1) for i in range(1, lanes)]


def run_lanes(cfg, lanes, gpus=None):
    """lanes: danh sách nhóm model, vd. [["ssl_aasist"], ["trihead_v2", ...]]. Lane nào lỗi thì
    vẫn gộp phần đã train rồi báo lỗi. gpus: số GPU (mặc định torch.cuda.device_count())."""
    if gpus is None:
        import torch

        gpus = max(1, torch.cuda.device_count())
    devices = lane_devices(len(lanes), gpus)
    runs = Path(cfg["runs"])
    runs.mkdir(parents=True, exist_ok=True)
    folders = lane_dirs(cfg, len(lanes))
    env_path = os.pathsep.join(p for p in sys.path if p)
    procs, codes = [], []
    try:
        for i, (models, lane) in enumerate(zip(lanes, folders)):
            lane.mkdir(parents=True, exist_ok=True)
            # Model đang dở / đã xong từ phiên trước (PREVIOUS_RUN đã gộp vào runs) về đúng lane.
            for name in models:
                for seed in cfg["seeds"]:
                    folder = runs / f"{name}_seed{seed}"
                    if folder.is_dir() and not (lane / folder.name).exists():
                        shutil.move(str(folder), str(lane / folder.name))
            config = lane / "lane-config.json"
            # run_group: tên nhóm W&B / artifact giữ như chạy một lane (không có hậu tố -gpu<i>).
            sub = {**cfg, "models": list(models), "runs": str(lane), "run_group": runs.name}
            write_json(config, sub)
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(devices[i]),
                "PYTHONPATH": env_path,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "vn_av_df.training.lanes", str(config)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            reader = threading.Thread(target=relay, args=(proc, f"[L{i}/GPU{devices[i]}] "), daemon=True)
            reader.start()
            procs.append((proc, reader))
            print(f"[L{i}/GPU{devices[i]}] lane {i}: {', '.join(models)} → {lane}", flush=True)
        codes = [proc.wait() for proc, _ in procs]
        for _, reader in procs:
            reader.join()
    finally:
        for proc, _ in procs:
            if proc.poll() is None:
                proc.terminate()
        # Gộp cả khi một lane lỗi / bị ngắt: phần đã train (last.pt) vẫn nằm trong Output.
        merge_runs_from_lanes(runs, folders)
    failed = [i for i, code in enumerate(codes) if code != 0]
    if failed:
        raise RuntimeError(
            f"Lane lỗi: {failed} (mã {[codes[i] for i in failed]}); xem log [GPU<i>]"
        )
    return codes


def relay(proc, prefix):
    for line in proc.stdout:
        print(prefix + line, end="", flush=True)


def merge_runs_from_lanes(runs, folders):
    from vn_av_df.training.train import merge_runs

    present = [f for f in folders if f.is_dir()]
    if present:
        merge_runs(runs, present, move=True)
        for folder in present:
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    from vn_av_df.training.train import train_all

    print(train_all(read_json(Path(sys.argv[1]))), flush=True)
