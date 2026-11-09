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


def stage(roots, encoders, sample_ids, dest, threads=64, reserve_gb=20.0, log=print):
    """Chép rồi trả danh sách gốc mới = [bản local...] + [gốc cũ...]. Không đủ chỗ trên ổ đích (giữ lại
    `reserve_gb`) thì không chép, trả nguyên `roots` (train vẫn chạy, chỉ chậm như cũ)."""
    roots = [str(r) for r in roots]
    jobs, staged = plan(roots, encoders, sample_ids, dest)
    if not jobs:
        return roots
    with ThreadPoolExecutor(threads) as pool:  # stat song song (độ trễ từng file)
        need = sum(pool.map(lambda job: os.path.getsize(job[0]), jobs)) / 1e9
    Path(dest).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free / 1e9
    if need > free - reserve_gb:
        log(f"stage: cần {need:.1f} GB, ổ {dest} còn {free:.1f} GB → không chép, đọc thẳng từ gốc")
        return roots
    for folder in {d.parent for _, d in jobs}:
        folder.mkdir(parents=True, exist_ok=True)
    began = time.time()

    def copy(job):
        source, target = job
        if not (target.is_file() and target.stat().st_size == source.stat().st_size):
            tmp = target.with_name(target.name + ".part")
            shutil.copyfile(source, tmp)
            os.replace(tmp, target)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(copy, jobs))
    spent = time.time() - began
    log(f"stage: {len(jobs)} file, {need:.1f} GB → {dest} trong {spent / 60:.1f} phút ({need * 1000 / max(spent, 1e-6):.0f} MB/s)")
    return [str(p) for p in staged] + roots


def stage_cfg(cfg, encoders, rows, dest, threads=64, splits=("train", "validation"), log=print):
    """Chép cache bản gốc + mọi biến thể cho `encoders` (vd. prepare.encoders_for(model của phiên)), mẫu thuộc
    `splits`; sửa cfg["cache"], cfg["variants"] tại chỗ. Trả danh sách encoder đã chép."""
    encoders = sorted(set(encoders) - {"landmarks"} | {"avhubert"})
    ids = [r["sample_id"] for r in rows if r["split"] in splits]
    # Ổ thiếu chỗ thì phần sau bị bỏ (đọc thẳng gốc): trước hết 3 encoder có ở mọi biến thể (M3-R đọc cặp
    # sạch ↔ nén), bản gốc rồi từng biến thể; encoder chỉ có ở bản gốc (lipinc, wave: BL-V, BL-A) sau cùng.
    first = [e for e in encoders if e in VARIANT_ENCODERS]
    later = [e for e in encoders if e not in VARIANT_ENCODERS]
    cfg["cache"] = stage(cfg["cache"] if isinstance(cfg["cache"], list) else [cfg["cache"]], first, ids,
                         Path(dest) / "clean", threads, log=log)
    for name, roots in (cfg.get("variants") or {}).items():
        cfg["variants"][name] = stage(roots, first, ids, Path(dest) / name, threads, log=log)
    if later:
        cfg["cache"] = stage(cfg["cache"], later, ids, Path(dest) / "clean-extra", threads, log=log)
    return encoders
