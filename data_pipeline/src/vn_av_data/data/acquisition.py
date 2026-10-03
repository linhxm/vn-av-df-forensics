"""Download only user-selected YouTube URLs, or index already downloaded source videos."""

from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vn_av_data.common.runtime import read_json, sha, write_json
from vn_av_data.data.manifest import read_manifest, write_manifest
from vn_av_data.data.media import MAX_SHORT_SIDE, MIN_FPS, ffmpeg, probe, quality_issue

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".mpg"}


def source_path(row, manifest):
    path = Path(row["video"])
    return path if path.is_absolute() else Path(manifest).resolve().parent / path


def youtube_id(url):
    import re

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        value = parsed.path.strip("/")
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        value = (
            parse_qs(parsed.query).get("v", [""])[0]
            if parsed.path == "/watch"
            else parsed.path.split("/")[-1]
            if parsed.path.startswith(("/shorts/", "/live/"))
            else ""
        )
    else:
        value = ""
    if parsed.scheme not in {"https", "http"} or not re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        raise ValueError(f"Use a single YouTube video URL: {url}")
    return value


# Luật chọn định dạng dùng chung cho bước 01 (xem trước, không tải) và 02 (tải thật).
# Chỉ luồng HTTPS: luồng HLS (vd. 616 Premium) tải tiếp có thể mất header mp4.
DOWNLOAD_FORMAT = (
    f"bestvideo[ext=mp4][protocol=https][fps>={MIN_FPS - 0.5}]"
    "+bestaudio[ext=m4a][protocol=https]"
    f"/best[ext=mp4][protocol=https][fps>={MIN_FPS - 0.5}]"
)
# res = cạnh ngắn: video 4K lấy bản 1080 có sẵn của YouTube (video dọc cũng đúng).
FORMAT_SORT = [f"res:{MAX_SHORT_SIDE}"]


def youtube_options(cookies_from_browser=None, force_ipv4=False):
    """Tuỳ chọn mạng chung của yt-dlp; cần Node.js để giải mã link YouTube."""
    import shutil

    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js must be on PATH for the YouTube downloader")
    options = {"socket_timeout": 30, "js_runtimes": {"node": {"path": node}}}
    if cookies_from_browser:
        options["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)
    if force_ipv4:
        options["source_address"] = "0.0.0.0"
    return options


def download_sources(
    selection, output, cookies_from_browser=None, force_ipv4=False, limit=0, dry_run=False
):
    import re

    import yt_dlp
    from yt_dlp.utils import PostProcessingError

    from vn_av_data.data.collect import normalize_sources
    from vn_av_data.data.source_io import read_rows, write_rows

    selected = normalize_sources(read_rows(selection))
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if dry_run:
        return {"selected": len(selected), "dry_run": True}
    network = youtube_options(cookies_from_browser, force_ipv4)
    output = Path(output).resolve()
    # Mỗi part: videos/ (media), sources.jsonl (manifest cho bước cắt), logs/ (chỉ để đọc).
    videos, logs = output / "videos", output / "logs"
    videos.mkdir(parents=True, exist_ok=True)
    snapshot = logs / "download_sources.csv"
    # Tải nhiều đợt: được nối thêm video mới, không được đổi/bỏ video đã có trong lượt tải.
    now = {r["video_id"]: r for r in selected}
    for row in read_rows(snapshot) if snapshot.exists() else []:
        new = now.get(row["video_id"])
        if new is None or {k: str(new.get(k, "") or "") for k in row} != row:
            raise FileExistsError(f"Selection changed for {row['video_id']}; use a new run")
    write_rows(snapshot, selected, mutable=True)
    manifest = output / "sources.jsonl"
    existing = (
        {row["source_id"]: row for row in read_manifest(manifest)} if manifest.exists() else {}
    )
    ids = {"yt_" + row["video_id"] for row in selected}
    if set(existing) - ids:
        raise ValueError("Run contains other sources; use a new download run")
    results = [{**row, "status": "pending", "filename": "", "error": ""} for row in selected]
    errors = []
    attempted = 0
    for result in results:
        video_id = result["video_id"]
        sid = "yt_" + video_id
        old = existing.get(sid)
        if old:
            path = source_path(old, manifest)
            if path.is_file() and sha(path) == old["sha256"]:
                metadata = probe(path)
                issue = quality_issue(metadata)
                if issue:
                    # Video tải trước khi có bước kiểm chất lượng: bỏ khỏi manifest, giữ file.
                    del existing[sid]
                    result.update(status="rejected", filename=path.name, error=issue)
                else:
                    existing[sid] = {**old, **metadata}
                    result.update(status="downloaded", filename=path.name)
                write_manifest(manifest, list(existing.values()))
                write_rows(logs / "download_results.csv", results, mutable=True)
                continue
            if path.is_file():
                # yt-dlp không ghi đè file có sẵn nên sẽ đăng ký nhầm file đã bị sửa.
                message = f"{path.name} changed after download; delete it to download again"
                result.update(status="failed", error=message)
                errors.append({"source_id": sid, "error": message})
                continue
        if limit and attempted >= limit:
            continue
        attempted += 1
        try:
            options = {
                **network,
                "format": DOWNLOAD_FORMAT,
                "format_sort": FORMAT_SORT,
                "outtmpl": str(videos / (video_id + ".%(ext)s")),
                "noplaylist": True,
                "merge_output_format": "mp4",
                "ffmpeg_location": ffmpeg(),
                # Tên file cố định theo video_id: bị ngắt thì lần sau tải tiếp từ file .part.
                "continuedl": True,
                "overwrites": False,
            }
            # Bước 01 đã loại video không đạt theo metadata; ở đây chỉ kiểm lại file thật.
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(result["url"], download=True)
            path = videos / (video_id + ".mp4")
            metadata = probe(path)
            issue = quality_issue(metadata)
            if issue:
                result.update(status="rejected", filename=path.name, error=issue)
                write_rows(logs / "download_results.csv", results, mutable=True)
                continue
            existing[sid] = {
                **{k: v for k, v in result.items() if k not in {"status", "filename", "error"}},
                "source_id": sid,
                "video": path.relative_to(output).as_posix(),
                "sha256": sha(path),
                "title": result.get("title") or info.get("title", ""),
                "channel": result.get("channel") or info.get("channel", ""),
                "speaker_id": result.get("speaker_id") or None,
                "dataset": "youtube",
                **metadata,
            }
            write_manifest(manifest, list(existing.values()))
            result.update(status="downloaded", filename=path.name)
        except Exception as exc:
            if isinstance(getattr(exc, "exc_info", (None, None))[1], PostProcessingError):
                # Ghép hỏng: xoá file từng luồng để lần sau tải lại từ đầu.
                for part in videos.glob(video_id + ".f*"):
                    part.unlink()
            message = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
            result.update(status="failed", error=message)
            errors.append({"source_id": sid, "error": message})
        write_rows(logs / "download_results.csv", results, mutable=True)
        write_json(logs / "download-errors.json", errors)
    write_rows(logs / "download_results.csv", results, mutable=True)
    if errors:
        raise RuntimeError(f"{len(errors)} downloads failed; rerun same batch to resume")
    return {
        "sources": len(existing),
        "rejected": {r["video_id"]: r["error"] for r in results if r["status"] == "rejected"},
        "pending": sum(r["status"] == "pending" for r in results),
        "manifest": str(manifest),
    }


def index_sources(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    rows, hashes = [], set()
    if output.exists():
        raise FileExistsError(f"Use a new source manifest: {output}")
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        digest = sha(path)
        if digest in hashes:
            continue
        hashes.add(digest)
        metadata_path = path.with_suffix(".source.json")
        meta = read_json(metadata_path) if metadata_path.exists() else {}
        import os

        rows.append(
            {
                **probe(path),
                "source_id": meta.get("source_id", "local_" + digest[:20]),
                "sha256": digest,
                "video": os.path.relpath(path, output.parent),
                "speaker_id": meta.get("speaker_id"),
                "dataset": meta.get("dataset", "local"),
            }
        )
    if not rows or len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("Need source videos with unique source IDs")
    write_manifest(output, rows)
    return {"sources": len(rows), "manifest": str(output)}
