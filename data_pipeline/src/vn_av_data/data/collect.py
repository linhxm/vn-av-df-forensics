# Adapted from the user-authorized VN-AV-DF-Capstone collect workflow.
"""Bước 01: bung playlist, chống trùng, xem metadata YouTube, kiểm chất lượng, chọn video.

Ba giai đoạn: (a) bung/chuẩn hoá/chống trùng; (b) hỏi metadata từng video (cần mạng, lưu
lại trong video_metadata.csv, lần sau không hỏi lại); (c) kiểm fps/độ phân giải/thời lượng
chỉ từ metadata (không cần mạng) rồi ghi selected_videos.csv với cột cố định.
"""

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vn_av_data.data.media import (
    MAX_SHORT_SIDE,
    MAX_SOURCE_HOURS,
    MIN_FPS,
    MIN_SHORT_SIDE,
    MIN_SOURCE_SECONDS,
)
from vn_av_data.data.source_io import read_rows, unique_rows, write_rows

# Danh sách để tải: chỉ thông tin ổn định (bước tải so snapshot từng dòng).
SELECTED_FIELDS = ["url", "video_id", "speaker_id", "title", "channel", "playlist_url"]
METADATA_FIELDS = [
    *("video_id", "url", "speaker_id", "playlist_url", "source_row"),
    *("title", "channel", "channel_id", "upload_date", "duration_s"),
    *("live_status", "availability"),
    # Bản gốc trên YouTube: định dạng hình lớn nhất và fps cao nhất người đăng cung cấp.
    *("max_width", "max_height", "native_fps", "vertical"),
    # Định dạng sẽ tải theo đúng luật của bước 02 (≤1080 cạnh ngắn, ≥25 fps, HTTPS mp4).
    *("v_format_id", "v_codec", "v_width", "v_height", "v_fps", "v_bitrate_kbps"),
    *("v_filesize_mb", "dynamic_range"),
    *("a_format_id", "a_codec", "a_bitrate_kbps", "a_sample_rate", "a_channels", "a_language"),
    *("audio_track_count", "audio_is_original"),
    *("status", "reason", "checked_at", "yt_dlp_version"),
]
# Mọi video không được chọn: trùng (same_part/other_part), bị loại (rejected), lỗi mạng (error).
SKIPPED_FIELDS = [
    *("video_id", "url", "title", "speaker_id", "source_row", "type", "reason"),
    *("kept_from", "kept_speaker_id", "speaker_conflict"),
]
# Lỗi từ YouTube cho biết video không xem được (loại hẳn); lỗi khác coi là lỗi mạng (hỏi lại).
UNAVAILABLE = re.compile(
    r"private|unavailable|removed|terminated|members|age|copyright|not available|deleted",
    re.IGNORECASE,
)


def video_id(value):
    value = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Expected a YouTube URL or video ID: {value}")
    if host in ("youtu.be", "www.youtu.be"):
        candidate = parsed.path.strip("/")
    elif host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
        parts = parsed.path.strip("/").split("/")
        candidate = parse_qs(parsed.query).get("v", [""])[0]
        if len(parts) == 2 and parts[0] in ("shorts", "embed", "live"):
            candidate = parts[1]
    else:
        raise ValueError(f"Expected a YouTube URL: {value}")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        raise ValueError(f"No valid video ID: {value}")
    return candidate


def normalize_sources(rows):
    output = []
    for row in rows:
        row = {key: (value or "").strip() for key, value in row.items() if key is not None}
        if not any(row.values()):
            continue
        vid = video_id(row.get("video_id") or row.get("url", ""))
        if row.get("url") and video_id(row["url"]) != vid:
            raise ValueError(f"video_id and URL disagree: {vid}")
        output.append(dict(row, video_id=vid, url="https://www.youtube.com/watch?v=" + vid))
    if not output:
        raise ValueError("Fill the url column in videos.csv before running this step")
    return list(unique_rows(output, "video_id").values())


def is_playlist(url):
    parsed = urlparse(url)
    return bool(
        parsed.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com")
        and parsed.path.rstrip("/") == "/playlist"
        and parse_qs(parsed.query).get("list")
    )


def expand_sources(rows, playlist_loader=None):
    """Mỗi video một dòng, kèm nơi xuất hiện (dòng CSV, playlist) để báo trùng; chưa bỏ trùng."""
    expanded = []
    for number, row in enumerate(rows, start=2):  # Dòng 1 của CSV là tiêu đề.
        row = {k: (v or "").strip() for k, v in row.items() if k is not None}
        url = row.get("url", "")
        if not any(row.values()):
            continue
        speaker = row.get("speaker_id", "")
        if not is_playlist(url):
            vid = video_id(row.get("video_id") or url)
            expanded.append(
                dict(video_id=vid, speaker_id=speaker, playlist_url="", source_row=f"row {number}")
            )
            continue
        if row.get("video_id"):
            raise ValueError("Leave video_id empty for a playlist row")
        if playlist_loader is None:
            import yt_dlp

            with yt_dlp.YoutubeDL({"extract_flat": "in_playlist", "skip_download": True}) as client:
                info = client.extract_info(url, download=False)
        else:
            info = playlist_loader(url)
        entries = list((info or {}).get("entries") or [])
        if not entries:
            raise ValueError(f"Playlist has no readable entries: {url}")
        for item in entries:
            if not item or not item.get("id"):
                raise ValueError(f"Unreadable playlist entry: {url}")
            expanded.append(
                dict(
                    video_id=video_id(item["id"]),
                    speaker_id=speaker,
                    playlist_url=url,
                    source_row=f"row {number} (playlist {url})",
                    title=item.get("title") or "",
                    channel=item.get("channel") or "",
                )
            )
    for item in expanded:
        item["url"] = "https://www.youtube.com/watch?v=" + item["video_id"]
    return expanded


def split_duplicates(items, taken):
    """Giữ lần xuất hiện đầu; bỏ lần sau và video đã thuộc part khác, ghi lại để sửa tay."""
    kept, duplicates = {}, []
    for item in items:
        first = kept.get(item["video_id"])
        part = taken.get(item["video_id"])
        if first is None and part is None:
            kept[item["video_id"]] = item
            continue
        duplicates.append(
            dict(
                video_id=item["video_id"],
                url=item["url"],
                title=item.get("title", ""),
                speaker_id=item["speaker_id"],
                source_row=item["source_row"],
                type="same_part" if first else "other_part",
                reason="duplicate" if first else "already selected in another part",
                kept_from=first["source_row"] if first else part,
                kept_speaker_id=first["speaker_id"] if first else "",
                speaker_conflict="yes"
                if first and first["speaker_id"] != item["speaker_id"]
                else "",
            )
        )
    return list(kept.values()), duplicates


def taken_videos(output):
    """video_id đã chọn ở các part khác, đọc từ thư mục anh em cùng tên file output."""
    output = Path(output).resolve()
    taken = {}
    for path in sorted(output.parent.parent.glob(f"*/{output.name}")):
        if path.parent == output.parent:
            continue
        for row in read_rows(path):
            if row.get("video_id") or row.get("url"):
                taken.setdefault(video_id(row.get("video_id") or row["url"]), path.parent.name)
    return taken


def youtube_metadata(url, options):
    """Một lần hỏi YouTube: metadata thô + định dạng bước 02 sẽ chọn (chọn offline, không tải)."""
    import copy

    import yt_dlp

    from vn_av_data.data.acquisition import DOWNLOAD_FORMAT, FORMAT_SORT

    quiet = {**options, "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(quiet) as client:
        raw = client.extract_info(url, download=False, process=False)
    selector = {**quiet, "format": DOWNLOAD_FORMAT, "format_sort": FORMAT_SORT}
    try:
        with yt_dlp.YoutubeDL(selector) as client:
            chosen = client.process_ie_result(copy.deepcopy(raw), download=False)
        raw["requested_formats"] = chosen.get("requested_formats") or [chosen]
    except yt_dlp.utils.YoutubeDLError:  # "Requested format is not available" là ExtractorError.
        raw["requested_formats"] = []  # Không có định dạng đạt; giai đoạn kiểm sẽ nêu lý do.
    return raw


def number(value, digits=1):
    return "" if value in (None, "") else round(float(value), digits)


def describe(info):
    """Dòng video_metadata.csv từ metadata yt-dlp (thô + requested_formats)."""
    formats = info.get("formats") or []
    video = [f for f in formats if f.get("vcodec") not in (None, "none")]
    audio = [
        f for f in formats if f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")
    ]
    largest = max(video, key=lambda f: min(f.get("width") or 0, f.get("height") or 0), default={})
    chosen = info.get("requested_formats") or []
    v = next((f for f in chosen if f.get("vcodec") not in (None, "none")), {})
    a = next((f for f in chosen if f.get("acodec") not in (None, "none")), {})
    languages = {f.get("language") for f in audio}
    note = (a.get("format_note") or "").lower()
    return {
        "title": info.get("title") or "",
        "channel": info.get("channel") or info.get("uploader") or "",
        "channel_id": info.get("channel_id") or "",
        "upload_date": info.get("upload_date") or "",
        "duration_s": number(info.get("duration")),
        "live_status": info.get("live_status") or "",
        "availability": info.get("availability") or "",
        "max_width": largest.get("width") or "",
        "max_height": largest.get("height") or "",
        "native_fps": number(max((f.get("fps") or 0 for f in video), default=0), 3),
        "vertical": "yes" if (largest.get("height") or 0) > (largest.get("width") or 0) else "",
        "v_format_id": v.get("format_id") or "",
        "v_codec": v.get("vcodec") or "",
        "v_width": v.get("width") or "",
        "v_height": v.get("height") or "",
        "v_fps": number(v.get("fps"), 3),
        "v_bitrate_kbps": number(v.get("vbr") or v.get("tbr")),
        "v_filesize_mb": number(
            size / 1e6 if (size := v.get("filesize") or v.get("filesize_approx")) else None
        ),
        "dynamic_range": v.get("dynamic_range") or "",
        "a_format_id": a.get("format_id") or "",
        "a_codec": a.get("acodec") or "",
        "a_bitrate_kbps": number(a.get("abr") or a.get("tbr")),
        "a_sample_rate": a.get("asr") or "",
        "a_channels": a.get("audio_channels") or "",
        "a_language": a.get("language") or "",
        # Nhiều track = có track lồng tiếng (vd. lồng tiếng AI của YouTube); kiểm track đã chọn.
        "audio_track_count": len(languages) if audio else "",
        "audio_is_original": "yes" if len(languages) <= 1 or "original" in note else "no",
    }


def check(row):
    """Kiểm chỉ từ metadata đã lưu; đổi ngưỡng thì chạy lại là ra danh sách mới, không cần mạng."""
    if row.get("live_status") in ("is_live", "is_upcoming", "post_live"):
        return f"live stream ({row['live_status']})"
    if row.get("availability") not in ("", "public", "unlisted"):
        return f"availability {row['availability']}"
    duration = float(row.get("duration_s") or 0)
    if not MIN_SOURCE_SECONDS <= duration <= MAX_SOURCE_HOURS * 3600:
        return f"duration {duration:.0f}s outside {MIN_SOURCE_SECONDS}s-{MAX_SOURCE_HOURS}h"
    # Dung sai 0,5: 25 fps danh định đôi khi ghi 24,99; 23,976 fps vẫn bị loại.
    if float(row.get("native_fps") or 0) < MIN_FPS - 0.5:
        return f"fps {float(row.get('native_fps') or 0):.2f} < {MIN_FPS}"
    if not row.get("v_format_id") or not row.get("a_format_id"):
        return f"no HTTPS mp4 format with ≥{MIN_FPS} fps and m4a audio"
    short = min(int(row["v_width"]), int(row["v_height"]))
    if short < MIN_SHORT_SIDE:
        return f"short side {short} px < {MIN_SHORT_SIDE}"
    if short > MAX_SHORT_SIDE:
        return f"no ≤{MAX_SHORT_SIDE} px format (short side {short})"
    return None


def collect_part(
    input_path,
    output,
    playlist_loader=None,
    metadata_loader=None,
    cookies_from_browser=None,
    force_ipv4=False,
):
    """Chọn video cho một part; chạy lại nhiều đợt, chỉ hỏi YouTube cho video mới hoặc lỗi mạng."""
    import yt_dlp

    output = Path(output)
    folder = output.parent
    taken = taken_videos(output)
    previous = read_rows(output) if output.exists() else []
    # Video đã chọn (có thể đã tải/cắt) ở part này mà part khác cũng có thì phải sửa tay.
    clash = sorted(r["video_id"] for r in previous if r["video_id"] in taken)
    if clash:
        raise ValueError(f"Videos already selected in another part: {clash}")
    items, duplicates = split_duplicates(
        expand_sources(read_rows(input_path), playlist_loader), taken
    )
    old = {r["video_id"]: r for r in previous}
    for item in items:
        before = old.get(item["video_id"])
        if before and (before.get("speaker_id") or "") != item["speaker_id"]:
            raise ValueError(
                f"Conflicting speaker_id for already selected {item['video_id']}; "
                "it may already be downloaded"
            )
    # (b) Metadata: dùng lại dòng đã hỏi, chỉ hỏi video mới hoặc lần trước lỗi mạng.
    metadata_path = folder / "video_metadata.csv"
    cache = {r["video_id"]: r for r in read_rows(metadata_path)} if metadata_path.exists() else {}
    if metadata_loader is None and any(
        cache.get(i["video_id"], {}).get("status") in (None, "error") for i in items
    ):
        from vn_av_data.data.acquisition import youtube_options

        network = youtube_options(cookies_from_browser, force_ipv4)

        def metadata_loader(url):
            return youtube_metadata(url, network)

    for index, item in enumerate(items, start=1):
        row = cache.get(item["video_id"])
        if row is None or row.get("status") == "error":
            print(f"Metadata {index}/{len(items)} {item['video_id']}", flush=True)
            try:
                row = describe(metadata_loader(item["url"]))
                row.update(status="", reason="")
            except Exception as exc:  # noqa: BLE001 -- ghi lại rồi chạy tiếp các video khác
                message = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
                # Tiền tố "youtube:" đánh dấu lý do do YouTube trả về, không kiểm lại đè lên.
                row = {"status": "rejected" if UNAVAILABLE.search(message) else "error"}
                row["reason"] = "youtube: " + message
            row.update(
                checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                yt_dlp_version=yt_dlp.version.__version__,
            )
        # Định danh lấy theo videos.csv hiện tại (đổi speaker_id thì cập nhật).
        row = {**row, **{k: item[k] for k in ("video_id", "url", "speaker_id", "playlist_url")}}
        row["source_row"] = item["source_row"]
        # (c) Kiểm lại mỗi lần chạy từ metadata đã lưu; không ghi đè lý do loại do YouTube.
        if row.get("status") != "error" and not row.get("reason", "").startswith("youtube:"):
            reason = check(row)
            row.update(status="rejected" if reason else "accepted", reason=reason or "")
        cache[item["video_id"]] = row
        current = [cache[i["video_id"]] for i in items if i["video_id"] in cache]
        write_rows(metadata_path, current, mutable=True, fields=METADATA_FIELDS)
    rows = [cache[i["video_id"]] for i in items]
    # Chỉ nối thêm: dòng cũ giữ nguyên thứ tự để lượt tải đang dở vẫn khớp snapshot.
    added = [
        {k: r.get(k, "") for k in SELECTED_FIELDS}
        for r in rows
        if r["status"] == "accepted" and r["video_id"] not in old
    ]
    selected = [{k: r.get(k, "") for k in SELECTED_FIELDS} for r in previous] + added
    if selected:
        write_rows(output, selected, mutable=True, fields=SELECTED_FIELDS)
    # Một file để xem nhanh mọi video bị bỏ và lý do; thông số đầy đủ ở video_metadata.csv.
    skipped = duplicates + [
        {k: r.get(k, "") for k in ("video_id", "url", "title", "speaker_id", "source_row")}
        | {"type": r["status"], "reason": r["reason"]}
        for r in rows
        if r["status"] in ("rejected", "error")
    ]
    report = folder / "skipped_videos.csv"
    if skipped:
        write_rows(report, skipped, mutable=True, fields=SKIPPED_FIELDS)
    else:
        report.unlink(missing_ok=True)
    for old_name in ("duplicates.csv", "cross_part_duplicates.csv"):  # Tên cũ, đã gộp.
        (folder / old_name).unlink(missing_ok=True)
    errors = [r["video_id"] for r in rows if r["status"] == "error"]
    summary = {
        "videos": len(rows),
        "accepted": sum(r["status"] == "accepted" for r in rows),
        "rejected": {r["video_id"]: r["reason"] for r in rows if r["status"] == "rejected"},
        "metadata_errors": errors,
        "selected_total": len(selected),
        "added": len(added),
        "duplicates_same_part": sum(d["type"] == "same_part" for d in duplicates),
        "duplicates_other_part": sum(d["type"] == "other_part" for d in duplicates),
        "speaker_conflicts": sum(d["speaker_conflict"] == "yes" for d in duplicates),
        "output": str(output),
    }
    if skipped:
        summary["note"] = f"See {report}; fix speaker_id in videos.csv if speaker_conflict"
    if errors:
        raise RuntimeError(f"Metadata failed for {len(errors)} videos (network?); rerun 01_collect")
    if not selected:
        raise ValueError("No video passed the quality check; see video_metadata.csv")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(collect_part(args.input, args.out))


if __name__ == "__main__":
    main()
