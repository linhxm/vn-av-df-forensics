"""Train các ARCHITECTURES/SEEDS đã chọn trong settings.py; không tự so sánh/test."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import RESUME, bootstrap, config

bootstrap()
from vn_av_df.actions import execute

if __name__ == "__main__":
    execute("train", config(), RESUME)
