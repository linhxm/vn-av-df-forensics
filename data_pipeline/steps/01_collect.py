import sys

sys.dont_write_bytecode = True
from _run import run
from settings import ROOT, SOURCES

if __name__ == "__main__":
    # Part mới: tự tạo bảng nguồn rỗng, không mang URL của part trước sang.
    if not (SOURCES / "videos.csv").exists():
        SOURCES.mkdir(parents=True, exist_ok=True)
        (SOURCES / "videos.csv").write_bytes((ROOT / "configs/videos.example.csv").read_bytes())
        raise SystemExit(f"Created {SOURCES / 'videos.csv'}. Fill URLs, then run again.")
    # Bung playlist, bỏ video trùng, hỏi metadata YouTube (lưu ở video_metadata.csv),
    # loại video <25 fps hoặc <720p, ghi selected_videos.csv để tải. Video trùng/bị loại/lỗi
    # mạng kèm lý do ghi ở skipped_videos.csv.
    # Chạy lại sau khi thêm URL: chỉ hỏi video mới. Có thể thêm --force-ipv4.
    run(["collect", "--input", SOURCES / "videos.csv", "--output", SOURCES / "selected_videos.csv"])
