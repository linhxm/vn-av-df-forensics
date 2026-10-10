"""Local review and binary demo; no explanation or manipulation-type predictions."""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from vn_av_data.serving.review_page import review_page

from vn_av_df.common.runtime import write_json
from vn_av_df.data.groups import read_manifest
from vn_av_df.generation import csv_write


def demo_app(cfg, analyzer_factory=None):
    root = Path(cfg["demo_output"])
    root.mkdir(parents=True, exist_ok=True)
    executor = ThreadPoolExecutor(max_workers=1)
    slots = threading.BoundedSemaphore(2)
    jobs = {}
    analyzer = []

    @asynccontextmanager
    async def lifespan(app):
        yield
        executor.shutdown(wait=True)

    app = FastAPI(lifespan=lifespan)

    def process(sid, path):
        try:
            jobs[sid]["status"] = "processing"
            if not analyzer:
                from vn_av_df.inference import Analyzer

                # Demo cần điểm theo ô để vẽ timeline AI và timeline từng nhánh bằng chứng P2.
                analyzer.append((analyzer_factory or Analyzer)({**cfg, "demo_details": True}))
            result = analyzer[0].analyze(path, root / sid / "result.json")
            jobs[sid].update(status="complete", result=result)
        except Exception as exc:
            jobs[sid].update(status="failed", error=str(exc))
        finally:
            try:
                write_json(root / sid / "job.json", jobs[sid])
            finally:
                slots.release()

    @app.get("/api/health")
    def health():
        from vn_av_df.inference import chosen_checkpoint

        try:
            ready = chosen_checkpoint(cfg).is_file()
        except (FileNotFoundError, KeyError, ValueError):
            ready = False
        return {
            "ready": ready,
            "message": "Checkpoint available; full assets checked on first analysis"
            if ready
            else "Train binary model or configure CHECKPOINT first",
        }

    @app.get("/api/research")
    def research():
        """Kết quả test từng detector của run đang chọn (evaluation.json); chưa test thì None."""
        from vn_av_df.common.runtime import read_json

        path = Path(cfg.get("runs", "runs")) / "evaluation.json"
        return {"runs": read_json(path)["runs"] if path.is_file() else None}

    @app.post("/api/jobs")
    async def submit(video: UploadFile = File(...)):
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "Queue full")
        sid = uuid.uuid4().hex
        folder = root / sid
        path = folder / "input.mp4"
        total = 0
        try:
            folder.mkdir()
            with path.open("wb") as stream:
                while chunk := await video.read(1024 * 1024):
                    total += len(chunk)
                    if total > 512 * 1024 * 1024:
                        raise HTTPException(413, "Maximum upload 512 MB")
                    stream.write(chunk)
            if not total:
                raise HTTPException(400, "Empty upload")
            jobs[sid] = {"id": sid, "status": "queued", "filename": video.filename}
            executor.submit(process, sid, path)
            return jobs[sid]
        except Exception:
            try:
                path.unlink(missing_ok=True)
                if folder.exists():
                    folder.rmdir()
                jobs.pop(sid, None)
            finally:
                slots.release()
            raise
        finally:
            await video.close()

    def get(sid):
        if sid not in jobs:
            raise HTTPException(404, "Job not found")
        return jobs[sid]

    @app.get("/api/jobs/{sid}")
    def status(sid: str):
        return get(sid)

    @app.get("/api/jobs/{sid}/media")
    def media(sid: str):
        get(sid)
        return FileResponse(root / sid / "input.mp4", media_type="video/mp4")

    @app.get("/api/jobs/{sid}/report")
    def report(sid: str):
        if get(sid)["status"] != "complete":
            raise HTTPException(409, "Not complete")
        return FileResponse(root / sid / "result.json", filename="result.json")

    # Đăng ký sau API để static frontend không che các route upload/report.
    frontend = Path(__file__).resolve().parents[2] / "demo" / "dist"
    if frontend.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


class Decision(BaseModel):
    decision: str


def review_app(cfg):
    import csv

    root = Path(cfg["generated_dataset"])
    rows = read_manifest(root / "candidates.jsonl")
    indexed = {r["sample_id"]: r for r in rows}
    with (root / "review.csv").open(encoding="utf-8-sig", newline="") as f:
        reviews = list(csv.DictReader(f))
    decisions = {r["sample_id"]: r["decision"] for r in reviews}
    lock = threading.Lock()
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def page():
        return REVIEW_PAGE

    @app.get("/items")
    def items():
        return [{**r, "decision": decisions[r["sample_id"]]} for r in rows]

    @app.get("/media/{sid}")
    def media(sid: str):
        if sid not in indexed:
            raise HTTPException(404)
        from vn_av_df.dataset import media_path

        return FileResponse(media_path(root, indexed[sid]), media_type="video/mp4")

    @app.post("/decision/{sid}")
    def decide(sid: str, body: Decision):
        if (root / "manifest.jsonl").exists():
            raise HTTPException(409, "Dataset finalized; create a new version")
        if sid not in indexed or body.decision not in {"keep", "reject", "uncertain"}:
            raise HTTPException(422)
        with lock:
            decisions[sid] = body.decision
            csv_write(
                root / "review.csv",
                [
                    {"sample_id": r["sample_id"], "decision": decisions[r["sample_id"]]}
                    for r in rows
                ],
            )
        return {"saved": True}

    return app


# Mẫu sinh mặc định keep: chỉ đánh reject/uncertain mẫu lỗi. Real gốc cùng clip chiếu cạnh để đối chiếu.
REVIEW_PAGE = review_page(
    "Duyệt dữ liệu Real/Fake",
    "Keep: media hoạt động, thấy rõ người nói, thao tác sinh đúng khoảng ghi nhận (vạch đỏ: đoạn fake, "
    "vạch cam: đoạn lệch tiếng của sham). Không chọn dựa trên điểm detector; đây là duyệt chất lượng "
    "dữ liệu, không phải đầu ra model.",
    items="/items",
    media="/media/{key}",
    save="/decision/{key}",
    key="sample_id",
    filters=["decision", "split", "variant", "generator", "audio_mode", "speaker_id"],
    tags=["variant", "generator"],
    info=[
        "sample_id",
        "split",
        "variant",
        "label",
        "generator",
        "audio_mode",
        "speaker_id",
        "source_clip_id",
        "duration_s",
        "fake_intervals",
        "av_mismatch_intervals",
        "audio_source_id",
        "generator_version",
        "decision",
    ],
    search=["sample_id", "source_clip_id", "speaker_id", "source_id"],
    pending=["uncertain", "pending"],
    timeline={"fake_intervals": "Đoạn fake", "av_mismatch_intervals": "Đoạn lệch tiếng (sham)"},
    pair={"match": "source_clip_id", "where": {"variant": "real"}},
)
