import sys

sys.dont_write_bytecode = True
from _run import run
from settings import RAW, SOURCES

if __name__ == "__main__":
    run(["download", "--sources", SOURCES / "selected_videos.csv", "--output", RAW])
