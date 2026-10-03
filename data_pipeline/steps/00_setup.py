"""Run once to download YuNet/Silero models for cutting clean data."""

import sys

sys.dont_write_bytecode = True
from _run import run

if __name__ == "__main__":
    run(["setup"])
