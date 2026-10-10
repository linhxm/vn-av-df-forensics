import sys

sys.dont_write_bytecode = True
import json

from _run import run
from settings import CUT, DATASET_VERSION, EXPORT, PART, SPLIT_HISTORY, SPLIT_RATIOS, SPLIT_SEED

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
            "--part",
            PART,
            "--split-seed",
            SPLIT_SEED,
            "--split-ratios",
            json.dumps(SPLIT_RATIOS),
            "--split-history",
            *SPLIT_HISTORY,
        ]
    )
