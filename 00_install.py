"""Chạy trong môi trường Anaconda chính đang active để cài thư viện.

PROFILE = "demo" tương đương: python -m pip install -e ./data_pipeline -e ".[training]"
Demo dùng chung thư viện inference với training, không chạy huấn luyện.
Script này không tải checkpoint, không cài worker AV-HuBERT riêng hoặc frontend.
Tải AV-HuBERT: python training/00_setup.py --encoder avhubert
"""

import subprocess
import sys
from pathlib import Path

# "data": thu thập/review; "demo": demo local.
# "generation" và "training": các profile dùng trên Kaggle (notebook đã tự cài).
PROFILE = "demo" 

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    if PROFILE not in {"data", "demo", "generation", "training"}:
        raise ValueError("PROFILE must be data, demo, generation or training")
    args = [sys.executable, "-m", "pip", "install", "-e", str(root / "data_pipeline")]
    extra = "training" if PROFILE == "demo" else PROFILE
    # Cài cả package chính cho review/finalize; demo lấy thêm dependency inference.
    args += ["-e", str(root) if extra == "data" else f"{root}[{extra}]"]
    subprocess.run(args, check=True)
