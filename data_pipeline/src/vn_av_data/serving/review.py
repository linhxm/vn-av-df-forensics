"""Local review of candidate clips. Decisions persist to the explicit review CSV."""

import csv
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from vn_av_data.data.curation import write_csv
from vn_av_data.serving.review_page import review_page

# Clip ứng viên từ bước cắt: mặc định "uncertain" = chưa duyệt; giữ khi một người, miệng rõ, tiếng khớp.
PAGE = review_page(
    "Duyệt clip môi-tiếng",
    "Giữ = một người, miệng thấy rõ, âm thanh thuộc người trong hình và khớp thời gian. "
    "Chưa chắc thì chọn Chưa rõ. Nguồn YouTube thật chưa tự bảo đảm đồng bộ.",
    items="/api/clips",
    media="/api/media/{key}",
    save="/api/clips/{key}",
    key="index",
    filters=["decision", "speaker_id", "source_id", "quality"],
    tags=["speaker_id"],
    info=[
        "clip_id",
        "speaker_id",
        "source_id",
        "channel",
        "source_start_s",
        "source_end_s",
        "duration_s",
        "face_ratio",
        "visual_sample_coverage",
        "source_fps",
        "source_width",
        "source_height",
        "source_codec",
        "source_bitrate_kbps",
        "quality",
        "decision",
        "url",
    ],
    search=["clip_id", "source_id", "speaker_id", "channel"],
    pending=["uncertain", "pending", ""],
)


class Decision(BaseModel):
    decision: str


def create_review_app(review, root):
    review, root = Path(review).resolve(), Path(root).resolve()
    with review.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    paths = []
    for row in rows:
        path = root / row["file_path"].replace("\\", "/")
        if not path.is_file():
            path = root / row["file_path"].replace("\\", "/").rsplit("/", 1)[-1]
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Review media missing/outside root: {path}")
        paths.append(path)
    app, lock = FastAPI(), threading.Lock()

    def check(index):
        if not 0 <= index < len(rows):
            raise HTTPException(404, "Clip not found")

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE

    @app.get("/api/clips")
    def clips():
        return rows

    @app.get("/api/media/{index}")
    def media(index: int):
        check(index)
        return FileResponse(paths[index])

    @app.post("/api/clips/{index}")
    def update(index: int, value: Decision):
        check(index)
        if value.decision not in {"keep", "reject", "uncertain"}:
            raise HTTPException(422, "Invalid decision")
        with lock:
            rows[index]["decision"] = value.decision
            if "sync_status" in rows[index]:
                rows[index]["sync_status"] = (
                    "reviewed_match" if value.decision == "keep" else "unverified"
                )
            write_csv(review, rows)
        return {"saved": True}

    return app
