"""Gán nhãn mismatch v004, in bảng đếm 0 / 1 / mask theo ô / generator + kiểm lối tắt độ trễ.

CPU, chạy sau prepare và trước train → runs/<RUN>/sync_labels.json.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import bootstrap

bootstrap()
from vn_av_df.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["sync-labels", *sys.argv[1:]]))
