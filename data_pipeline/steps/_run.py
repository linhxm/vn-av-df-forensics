import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from settings import DATASET_VERSION, ROOT

sys.path.insert(0, str(ROOT / "src"))
from vn_av_data.cli import main


def run(args):
    """Hiện part/bước đang chạy để tránh điền nhầm CSV hoặc xử lý nhầm part."""
    print(f"Part: {DATASET_VERSION} | Step: {args[0]}", flush=True)
    os.chdir(ROOT)
    raise SystemExit(main([str(x) for x in args] + sys.argv[1:]))
