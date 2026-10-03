import sys

sys.dont_write_bytecode = True
from _run import run
from settings import CUT, DATASET_VERSION, EXPORT

if __name__ == "__main__":
    run(
        [
            "export",
            "--root",
            CUT,
            "--review",
            CUT / "review.csv",
            "--output",
            EXPORT,
            "--dataset-id",
            DATASET_VERSION,
        ]
    )
