"""Bản part 1 cắt 0,4 s cuối mọi sample (real lẫn fake) để xoá lối tắt "khoảng lặng cuối" của
② sham full / ⑤ donor full (part 1: AUC 0,08 / 0,10; cắt 0,4 s: 0,48 / 0,47).
Chỉ dùng cho part 1 đã sinh trước khi có `donor_fit` (23/10); part sau sinh đúng từ đầu.

KHÔNG sửa thư mục gốc: đọc <src>, ghi bản mới vào <dst> (thư mục mới). Mỗi sample mã hoá lại
đúng cấu hình lúc sinh (H.264 CRF 18 + yuv420p, AAC 48 kHz mono) và bỏ 10 frame / 0,4 s cuối;
mọi sample qua cùng một bước nên không tạo khác biệt giữa real và fake. Nhãn tính lại bằng
cells.labels (đoạn partial luôn cách mép ≥ 0,5 s nên không mất nhãn), sha256 / độ dài mới,
thêm trường trim_end_s. Cuối cùng validate_rows + audit lối tắt (shortcut_audit.json).
Chạy lại thì bỏ qua file đã xong (ghi tạm rồi đổi tên).

    python tools/trim_part1.py . <src part dir> <dst part dir> [--workers 4] [--limit N]

Đã chạy 23/10: vn-av-df-data-part1_fake → vn-av-df-data-part1_trim04_fake (7.622 file, 1,4 h
CPU 16 luồng, 0 cờ lối tắt); bản này là dataset Kaggle `vn-av-df-data-part1-trim04-fake`.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TRIM_FRAMES = 10  # 0,4 s ở 25 fps
SPF = 1920  # mẫu 48 kHz mỗi frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="chỉ làm N sample đầu (chạy thử)")
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args()
    sys.path[:0] = [os.path.join(args.repo, "src")]
    from vn_av_df.common.runtime import sha, write_json
    from vn_av_df.data.groups import write_manifest
    from vn_av_df.data.media import ffmpeg, probe
    from vn_av_df.dataset import validate_rows
    from vn_av_df.generation.cells import labels

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if dst == src or src in dst.parents or dst in src.parents:
        raise SystemExit("dst phải là thư mục khác, không lồng với src")
    rows = [
        json.loads(x) for x in (src / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if args.limit:
        rows = rows[: args.limit]
    (dst / "clips").mkdir(parents=True, exist_ok=True)
    exe = ffmpeg()

    def trim(row):
        frames = round(row["duration_s"] * 25)
        keep = frames - TRIM_FRAMES
        target = dst / row["video"]
        if not target.exists():
            partial = target.with_name(target.stem + ".partial.mp4")
            subprocess.run(
                [
                    exe,
                    *("-nostdin", "-v", "error", "-y", "-threads", "4", "-i", src / row["video"]),
                    *("-map", "0:v:0", "-map", "0:a:0", "-frames:v", str(keep)),
                    *("-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-crf", "18"),
                    *("-pix_fmt", "yuv420p"),
                    *(
                        "-af",
                        f"aresample=48000,atrim=end_sample={keep * SPF},apad=whole_len={keep * SPF}",
                    ),
                    *("-c:a", "aac", "-ar", "48000", "-ac", "1", partial),
                ],
                check=True,
                capture_output=True,
            )
            os.replace(partial, target)
        info = probe(target)
        if abs(info["duration_s"] - keep / 25) > 0.1:
            raise ValueError(f"{row['sample_id']}: duration {info['duration_s']} != {keep / 25}")
        duration = keep / 25
        span = row.get("partial_span")
        if span and span[1] > duration - 1e-6:
            raise ValueError(f"{row['sample_id']}: partial span {span} cut by trim")
        return {
            **row,
            "sha256": sha(target),
            "duration_s": duration,
            "trim_end_s": TRIM_FRAMES / 25,
            **labels(row["cell"], row["kind"], duration, span),
        }

    began, out = time.time(), {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(trim, r): r["sample_id"] for r in rows}
        for i, future in enumerate(as_completed(futures), 1):
            out[futures[future]] = future.result()
            if i % 100 == 0 or i == len(rows):
                left = (time.time() - began) / i * (len(rows) - i)
                print(f"{i}/{len(rows)} sample, còn ~{left / 60:.0f} phút", flush=True)
    kept = [out[r["sample_id"]] for r in rows]
    validate_rows(kept, dst, verify_media=True)
    write_manifest(dst / "manifest.jsonl", kept)
    info = json.loads((src / "dataset_info.json").read_text(encoding="utf-8"))
    info.update(
        samples=len(kept),
        manifest_sha256=sha(dst / "manifest.jsonl"),
        trim_end_s=TRIM_FRAMES / 25,
        trimmed_from_manifest_sha256=info["manifest_sha256"],
    )
    write_json(dst / "dataset_info.json", info)
    if not args.no_audit:
        from vn_av_df.generation.audit import audit

        report = audit(dst, kept)
        write_json(dst / "shortcut_audit.json", report)
        print("cờ lối tắt:", report["flags"] or "không có")
        for group in ("sham/full", "donor/full"):
            if group in report["by_cell_vs_real"]:
                print(
                    group,
                    "AUC khoảng lặng cuối:",
                    round(report["by_cell_vs_real"][group]["trail_silence_s"], 3),
                )
    print("xong:", dst)


if __name__ == "__main__":
    main()
