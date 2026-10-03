import sys

sys.dont_write_bytecode = True
from _run import run
from settings import CUT

if __name__ == "__main__":
    # Duyệt trực tiếp review.csv của part; clip mới cắt thêm được nối vào cuối file này.
    run(["review", "--root", CUT, "--review", CUT / "review.csv"])
