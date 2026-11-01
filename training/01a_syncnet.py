"""(Local, CPU) Đo đường cong SyncNet cho real + lip-sync full → <part>/sync_syncnet.json.

Nhãn mismatch v004 (training/sync_labels.py). Chạy trước khi upload dataset lên Kaggle; dừng
giữa chừng được. --workers N --threads N --limit N.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import bootstrap

bootstrap()
from vn_av_df.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["syncnet", *sys.argv[1:]]))
