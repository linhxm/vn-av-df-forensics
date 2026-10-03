from pathlib import Path

import pytest

from vn_av_data.data.acquisition import download_sources
from vn_av_data.data.collect import (
    SELECTED_FIELDS,
    collect_part,
    expand_sources,
    split_duplicates,
)
from vn_av_data.data.curation import curate_sources
from vn_av_data.data.manifest import read_manifest
from vn_av_data.data.source_io import read_rows, write_rows

GOOD = {"duration_s": 5, "fps": 30.0, "width": 1920, "height": 1080}


def youtube_info(fps=30, width=1920, height=1080, duration=600, dubbed=False, title="T"):
    """Metadata giả lập youtube_metadata(): định dạng thô + định dạng bước 02 sẽ chọn."""
    video = {"format_id": "399", "vcodec": "av01", "ext": "mp4", "protocol": "https"}
    video.update(fps=fps, width=width, height=height, vbr=900, filesize=5e7)
    audio = {"format_id": "140", "vcodec": "none", "acodec": "mp4a.40.2", "ext": "m4a"}
    audio.update(abr=128, asr=44100, audio_channels=2, language="vi")
    audio["format_note"] = "Vietnamese original (default)" if dubbed else "medium"
    formats = [video, audio]
    if dubbed:
        formats.append({**audio, "format_id": "140-1", "language": "en"})
    return {
        "title": title,
        "channel": "C",
        "duration": duration,
        "live_status": "not_live",
        "availability": "public",
        "formats": formats,
        "requested_formats": [video, audio] if fps >= 24.5 else [],
    }


def read_rows_manifest(folder):
    return read_manifest(folder / "sources.jsonl")


def test_expand_and_duplicates_report_origin_and_speaker_conflict():
    rows = [
        {"url": "https://youtube.com/playlist?list=one", "speaker_id": "person1"},
        {"url": "https://youtu.be/abcdefghijk", "speaker_id": "person2"},
        {"url": "https://youtube.com/watch?v=lmnopqrstuv&list=one", "speaker_id": "person3"},
    ]
    items = expand_sources(rows, lambda _: {"entries": [{"id": "abcdefghijk", "title": "A"}]})
    assert [i["source_row"] for i in items] == [
        "row 2 (playlist https://youtube.com/playlist?list=one)",
        "row 3",
        "row 4",
    ]
    kept, duplicates = split_duplicates(items, {"lmnopqrstuv": "part0"})
    assert [k["video_id"] for k in kept] == ["abcdefghijk"]
    same, other = duplicates
    # Giữ lần đầu (trong playlist), báo dòng 3 trùng và khác speaker để sửa tay.
    assert same["type"] == "same_part" and same["speaker_conflict"] == "yes"
    assert same["kept_from"].startswith("row 2 (playlist")
    assert (same["kept_speaker_id"], same["speaker_id"]) == ("person1", "person2")
    assert (other["type"], other["kept_from"]) == ("other_part", "part0")


def test_download_snapshot_resume_pending_and_cut_gate(tmp_path, monkeypatch):
    from vn_av_data.data import acquisition

    selected = tmp_path / "selected.csv"
    out = tmp_path / "raw"
    rows = [
        {"url": "https://youtu.be/abcdefghijk", "speaker_id": "one"},
        {"url": "https://youtu.be/lmnopqrstuv", "speaker_id": "two"},
    ]
    write_rows(selected, rows)
    assert download_sources(selected, out, dry_run=True)["selected"] == 2
    assert not out.exists()
    monkeypatch.setattr("shutil.which", lambda _: "node")
    monkeypatch.setattr(acquisition, "probe", lambda _: GOOD)
    requests = []

    class Downloader:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def extract_info(self, url, download=True):
            requests.append(url)
            Path(self.opts["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(url.encode())
            assert self.opts["js_runtimes"]["node"]["path"] == "node"
            return {"title": "fixture"}

    monkeypatch.setattr("yt_dlp.YoutubeDL", Downloader)
    report = download_sources(selected, out, limit=1)
    assert report["pending"] == 1
    with pytest.raises(ValueError, match="pending/failed"):
        curate_sources(out / "sources.jsonl", tmp_path / "cut", {})
    download_sources(selected, out)
    assert len(requests) == 2
    assert {p.name for p in out.iterdir()} == {"videos", "logs", "sources.jsonl"}
    assert read_rows_manifest(out)[0]["video"] == "videos/abcdefghijk.mp4"
    assert all(r["status"] == "downloaded" for r in read_rows(out / "logs/download_results.csv"))
    download_sources(selected, out)
    assert len(requests) == 2
    # Đợt sau được nối thêm video; chỉ video mới được tải.
    rows.append({"url": "https://youtu.be/wxyzabcdefg", "speaker_id": "three"})
    write_rows(selected, rows, mutable=True)
    download_sources(selected, out)
    assert requests[2:] == ["https://www.youtube.com/watch?v=wxyzabcdefg"]
    (out / "videos/wxyzabcdefg.mp4").write_bytes(b"edited")
    with pytest.raises(RuntimeError, match="1 downloads failed"):
        download_sources(selected, out)
    assert len(requests) == 3
    assert "changed after download" in read_rows(out / "logs/download_results.csv")[2]["error"]
    rows[0]["speaker_id"] = "changed"
    write_rows(selected, rows, mutable=True)
    with pytest.raises(FileExistsError):
        download_sources(selected, out)


def test_collect_part_metadata_quality_check_and_fixed_columns(tmp_path):
    root = tmp_path / "sources"
    part1, part2 = root / "part1", root / "part2"
    part1.mkdir(parents=True)
    part2.mkdir()
    write_rows(part1 / "selected_videos.csv", [{"video_id": "aaaaaaaaaaa", "url": ""}])
    infos = {
        "bbbbbbbbbbb": youtube_info(dubbed=True),  # Hợp lệ; có track lồng tiếng.
        "ccccccccccc": youtube_info(fps=23.976),  # 24 fps: loại.
        "ddddddddddd": youtube_info(width=854, height=480),  # 480p: loại.
        "eeeeeeeeeee": youtube_info(width=1080, height=1920),  # Video dọc, cạnh ngắn 1080.
    }
    asked = []

    def loader(url):
        asked.append(url)
        if url.endswith("fffffffffff") and "fffffffffff" not in infos:
            raise RuntimeError("Connection timed out")
        if url.endswith("ggggggggggg"):
            raise RuntimeError("ERROR: [youtube] ggggggggggg: Private video")
        return infos[url[-11:]]

    ids = ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc", "ddddddddddd", "eeeeeeeeeee"]
    ids += ["fffffffffff", "ggggggggggg", "bbbbbbbbbbb"]
    rows = [{"url": "https://youtu.be/" + v, "speaker_id": "p"} for v in ids]
    write_rows(part2 / "videos.csv", rows)
    output = part2 / "selected_videos.csv"
    with pytest.raises(RuntimeError, match="Metadata failed for 1 videos"):
        collect_part(part2 / "videos.csv", output, metadata_loader=loader)
    # Video lẻ cũng có title/channel; cột cố định dù có playlist hay không.
    selected = read_rows(output)
    assert list(selected[0]) == SELECTED_FIELDS
    assert [r["video_id"] for r in selected] == ["bbbbbbbbbbb", "eeeeeeeeeee"]
    assert (selected[0]["title"], selected[0]["playlist_url"]) == ("T", "")
    meta = {r["video_id"]: r for r in read_rows(part2 / "video_metadata.csv")}
    assert meta["ccccccccccc"]["reason"] == "fps 23.98 < 25"
    assert meta["ddddddddddd"]["reason"] == "short side 480 px < 720"
    assert meta["ggggggggggg"]["status"] == "rejected"
    assert meta["fffffffffff"]["status"] == "error"
    track = meta["bbbbbbbbbbb"]
    assert (track["audio_track_count"], track["audio_is_original"]) == ("2", "yes")
    # Một file liệt kê mọi video bị bỏ: trùng, bị loại (kèm lý do) và lỗi mạng.
    skipped = {(d["video_id"], d["type"]) for d in read_rows(part2 / "skipped_videos.csv")}
    assert {("aaaaaaaaaaa", "other_part"), ("bbbbbbbbbbb", "same_part")} <= skipped
    assert {("ccccccccccc", "rejected"), ("ggggggggggg", "rejected")} <= skipped
    assert ("fffffffffff", "error") in skipped
    # Chạy lại: chỉ hỏi lại video lỗi mạng; video đã có metadata không hỏi lại.
    asked.clear()
    infos["fffffffffff"] = youtube_info(title="F")
    report = collect_part(part2 / "videos.csv", output, metadata_loader=loader)
    assert asked == ["https://www.youtube.com/watch?v=fffffffffff"]
    assert report["added"] == 1
    assert [r["video_id"] for r in read_rows(output)][-1] == "fffffffffff"
    # Đổi speaker_id của video đã chọn (có thể đã tải) thì dừng.
    write_rows(part2 / "videos.csv", [{**rows[1], "speaker_id": "other"}], mutable=True)
    with pytest.raises(ValueError, match="Conflicting speaker_id"):
        collect_part(part2 / "videos.csv", output, metadata_loader=loader)
    write_rows(part1 / "selected_videos.csv", [{"video_id": "bbbbbbbbbbb"}], mutable=True)
    with pytest.raises(ValueError, match="another part"):
        collect_part(part2 / "videos.csv", output, metadata_loader=loader)


def test_download_postprocessing_failure_removes_streams(tmp_path, monkeypatch):
    from yt_dlp.utils import DownloadError, PostProcessingError

    from vn_av_data.data import acquisition

    selected = tmp_path / "selected.csv"
    out = tmp_path / "raw"
    write_rows(selected, [{"url": "https://youtu.be/abcdefghijk", "speaker_id": "one"}])
    monkeypatch.setattr("shutil.which", lambda _: "node")
    monkeypatch.setattr(acquisition, "probe", lambda _: GOOD)

    class Downloader:
        def __init__(self, opts):
            assert "protocol=https" in opts.get("format", "protocol=https")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def extract_info(self, url, download=True):
            (out / "videos/abcdefghijk.f616.mp4").write_bytes(b"broken")
            (out / "videos/abcdefghijk.f140.m4a").write_bytes(b"audio")
            cause = PostProcessingError("Invalid data")
            raise DownloadError(
                "\x1b[0;31mERROR:\x1b[0m Postprocessing: Invalid data", (type(cause), cause, None)
            )

    monkeypatch.setattr("yt_dlp.YoutubeDL", Downloader)
    with pytest.raises(RuntimeError, match="1 downloads failed"):
        download_sources(selected, out)
    assert not list((out / "videos").glob("abcdefghijk.f*"))
    assert read_rows(out / "logs/download_results.csv")[0]["error"].startswith(
        "ERROR: Postprocessing"
    )


def test_download_rechecks_downloaded_file_quality(tmp_path, monkeypatch):
    from vn_av_data.data import acquisition

    selected = tmp_path / "selected.csv"
    out = tmp_path / "raw"
    rows = [
        {"url": "https://youtu.be/abcdefghijk", "speaker_id": "one"},
        {"url": "https://youtu.be/lmnopqrstuv", "speaker_id": "two"},
    ]
    write_rows(selected, rows)
    monkeypatch.setattr("shutil.which", lambda _: "node")
    meta = {"abcdefghijk": GOOD, "lmnopqrstuv": {**GOOD, "fps": 23.976}}
    monkeypatch.setattr(acquisition, "probe", lambda path: meta[Path(path).stem])

    class Downloader:
        def __init__(self, opts):
            self.opts = opts
            assert "[fps>=24.5]" in opts["format"] and opts["format_sort"] == ["res:1080"]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def extract_info(self, url, download=True):
            Path(self.opts["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(url.encode())
            return {"title": "fixture"}

    monkeypatch.setattr("yt_dlp.YoutubeDL", Downloader)
    report = download_sources(selected, out)
    # File thật không đạt (YouTube đổi sau bước 01): rejected, không tính là lỗi.
    assert report["sources"] == 1 and list(report["rejected"]) == ["lmnopqrstuv"]
    status = {r["video_id"]: r["status"] for r in read_rows(out / "logs/download_results.csv")}
    assert status == {"abcdefghijk": "downloaded", "lmnopqrstuv": "rejected"}
    assert read_rows_manifest(out)[0]["fps"] == 30.0
    meta["abcdefghijk"] = {**GOOD, "height": 480, "width": 854}
    assert download_sources(selected, out)["sources"] == 0
    assert (out / "videos/abcdefghijk.mp4").exists()
