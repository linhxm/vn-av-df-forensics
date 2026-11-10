"""Chép cache cần cho phiên train sang ổ local trước khi train (v005, RESEARCH_WORKLOG/v005_model.md §11).

Đo trên Kaggle 08/11/2026 (kernel test-io-cache-p3, part 3): đọc /kaggle/input bị chặn bởi độ trễ từng file
(~100 ms): 1 luồng 10 MB/s, 4 luồng 46, 16 luồng 169, 64 luồng 438 MB/s; /kaggle/temp đọc nguội 1 luồng
133 MB/s. M3 (v004) đọc ~3,1 MB / mẫu mỗi epoch với 2 worker → ~20 MB/s → 75-96 phút / epoch dù tính toán
chỉ ~15 phút. Chép một lần bằng nhiều luồng (~5 phút cho ~120 GB) rồi đọc từ ổ local.

Chỉ chép file .npz của mẫu cần (train + validation), của encoder model cần, cùng các file phụ của thư mục
(extraction.json, sync_profile.json, ...); sidecar .json từng mẫu không chép (train không đọc). Thư mục
chép đứng TRƯỚC thư mục gốc trong danh sách cache: mẫu chưa chép (test) vẫn đọc từ gốc."""

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VARIANT_ENCODERS = ("avhubert", "mouth", "xlsr")  # encoder trích trên bản biến thể (prepare)


def plan(roots, encoders, sample_ids, dest):
    """[(nguồn, đích)] và danh sách gốc mới (cùng thứ tự) cho các thư mục cache `roots`."""
    keep = set(sample_ids)
    jobs, staged = [], []
    for index, root in enumerate(Path(r) for r in roots):
        target = Path(dest) / f"{index:02d}-{root.parent.parent.parent.name or root.name}"
        found = False
        for encoder in encoders:
            folder = root / encoder
            if not folder.is_dir():
                continue
            found = True
            # Một lần liệt kê thư mục (is_file dùng d_type, không stat từng file: stat trên /kaggle/input ~100 ms).
            names = [e.name for e in os.scandir(folder) if e.is_file()]
            stems = {n[:-4] for n in names if n.endswith(".npz")}
            for name in names:
                if name.endswith(".npz"):
                    if name[:-4] not in keep:
                        continue
                elif name.endswith(".json") and name[:-5] in stems:
                    continue  # sidecar của từng mẫu
                jobs.append((folder / name, target / encoder / name))
        if found:
            staged.append(target)
    return jobs, staged


def _stage(roots, encoders, sample_ids, dest, threads=64, reserve_gb=20.0, budget_gb=None, log=print):
    """(danh sách gốc mới, GB đã chép). Chép theo mẫu (mọi encoder của một mẫu cùng chép hoặc cùng bỏ), thứ
    tự `sample_ids`, tới khi hết min(chỗ trống − `reserve_gb`, `budget_gb`); mẫu không chép đọc thẳng gốc."""
    roots = [str(r) for r in roots]
    jobs, staged = plan(roots, encoders, sample_ids, dest)
    if not jobs:
        return roots, 0.0
    with ThreadPoolExecutor(threads) as pool:  # stat song song (độ trễ từng file)
        sizes = list(pool.map(lambda job: os.path.getsize(job[0]), jobs))
    Path(dest).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free / 1e9
    allowed = free - reserve_gb if budget_gb is None else min(free - reserve_gb, budget_gb)
    need = sum(sizes) / 1e9
    by_sample = {}
    for job, size in zip(jobs, sizes):
        stem = job[0].name[:-4] if job[0].name.endswith(".npz") else None  # None: file phụ, luôn chép
        by_sample.setdefault(stem, []).append((job, size))
    chosen, used = [], 0.0
    for sid in [None, *dict.fromkeys(sample_ids)]:
        group = by_sample.get(sid, [])
        size = sum(s for _, s in group) / 1e9
        if used + size > allowed:
            break
        chosen += [job for job, _ in group]
        used += size
    if not any(job[0].name.endswith(".npz") for job in chosen):
        log(f"stage: cần {need:.1f} GB, được chép {allowed:.1f} GB (ổ {dest} còn {free:.1f} GB) → không chép, đọc thẳng từ gốc")
        return roots, 0.0
    for folder in {d.parent for _, d in chosen}:
        folder.mkdir(parents=True, exist_ok=True)
    began = time.time()

    def copy(job):
        source, target = job
        if not (target.is_file() and target.stat().st_size == source.stat().st_size):
            tmp = target.with_name(target.name + ".part")
            shutil.copyfile(source, tmp)
            os.replace(tmp, target)

    try:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(copy, chosen))
    except OSError as error:
        # 10/11 (train-p1): Errno 30 Read-only file system giữa chừng khi /kaggle/temp đã có ~95 GB dù
        # disk_usage báo còn chỗ. File chép xong vẫn dùng được (ghi .part rồi mới đổi tên); file thiếu đọc gốc.
        log(f"stage: CẢNH BÁO lỗi chép vào {dest} ({error.strerror}: {error.filename2 or error.filename}); "
            "phần chưa chép đọc thẳng từ gốc")
        return [str(p) for p in staged] + roots, used
    spent = time.time() - began
    part = "" if len(chosen) == len(jobs) else f" (một phần: {used:.1f} / {need:.1f} GB, trần {allowed:.1f} GB)"
    log(f"stage: {len(chosen)} file, {used:.1f} GB → {dest} trong {spent / 60:.1f} phút "
        f"({used * 1000 / max(spent, 1e-6):.0f} MB/s){part}")
    return [str(p) for p in staged] + roots, used


def stage(roots, encoders, sample_ids, dest, threads=64, reserve_gb=20.0, budget_gb=None, log=print):
    """Chép rồi trả danh sách gốc mới = [bản local...] + [gốc cũ...]. Không đủ chỗ trên ổ đích (giữ lại
    `reserve_gb`, tối đa `budget_gb`) thì chỉ chép một phần mẫu hoặc không chép, trả nguyên `roots` (train
    vẫn chạy, phần không chép chỉ chậm như cũ)."""
    return _stage(roots, encoders, sample_ids, dest, threads, reserve_gb, budget_gb, log)[0]


def stage_cfg(cfg, encoders, rows, dest, threads=64, splits=("train", "validation"), budget_gb=None, log=print):
    """Chép cache bản gốc + mọi biến thể cho `encoders` (vd. prepare.encoders_for(model của phiên)), mẫu thuộc
    `splits`; sửa cfg["cache"], cfg["variants"] tại chỗ. `budget_gb`: trần tổng GB chép. Trả danh sách encoder."""
    encoders = sorted(set(encoders) - {"landmarks"} | {"avhubert"})
    # Validation trước: hết trần thì phần thiếu rơi vào mẫu train (loader có worker đọc trước), không rơi vào
    # validation chấm mỗi epoch ở bản gốc + mọi mức nén.
    ids = [r["sample_id"] for split in sorted(splits, key=lambda s: s != "validation") for r in rows
           if r["split"] == split]
    # Hết trần thì phần sau bị bỏ (đọc thẳng gốc): trước hết 3 encoder có ở mọi biến thể (M3-R đọc cặp
    # sạch ↔ nén), bản gốc (mọi model đọc mỗi bước) rồi các biến thể chia đều phần còn lại (cùng mẫu ở mọi
    # biến thể); encoder chỉ có ở bản gốc (lipinc, wave: BL-V, BL-A) sau cùng.
    first = [e for e in encoders if e in VARIANT_ENCODERS]
    later = [e for e in encoders if e not in VARIANT_ENCODERS]
    left = float("inf") if budget_gb is None else float(budget_gb)
    cfg["cache"], used = _stage(cfg["cache"] if isinstance(cfg["cache"], list) else [cfg["cache"]], first, ids,
                                Path(dest) / "clean", threads, budget_gb=left, log=log)
    left -= used
    variants = list(cfg.get("variants") or {})
    for index, name in enumerate(variants):
        share = left / (len(variants) - index)
        cfg["variants"][name], used = _stage(cfg["variants"][name], first, ids, Path(dest) / name, threads,
                                             budget_gb=share, log=log)
        left -= used
    if later:
        cfg["cache"], used = _stage(cfg["cache"], later, ids, Path(dest) / "clean-extra", threads,
                                    budget_gb=left, log=log)
    return encoders
