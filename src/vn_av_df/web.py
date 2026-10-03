"""Local review and binary demo; no explanation or manipulation-type predictions."""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

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
                from vn_av_df.inference import Analyzer, ComparisonAnalyzer

                factory = ComparisonAnalyzer if cfg.get("demo_compare") else Analyzer
                # Demo cần điểm theo ô để vẽ timeline AI và timeline từng nhánh bằng chứng P2.
                analyzer.append((analyzer_factory or factory)({**cfg, "demo_details": True}))
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
        """Chỉ trả artifact thực; dataset trống thì không tạo số liệu minh họa."""
        from vn_av_df.common.runtime import read_json

        folder = Path(cfg.get("runs", "runs"))
        return {
            name: read_json(folder / name) if (folder / name).is_file() else None
            for name in ("comparison.json", "test-comparison.json")
        }

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


REVIEW_PAGE = """<!doctype html><meta charset="utf-8"><title>Review dataset</title>
<style>body{max-width:900px;margin:25px auto;font:16px system-ui}video{width:100%}button,select{padding:10px;margin:6px}</style>
<h1>Duyệt dữ liệu Real/Fake</h1><p>Keep: media hoạt động, thấy rõ người nói, thao tác sinh đúng khoảng ghi nhận. Không chọn dựa trên điểm detector. Đây là duyệt chất lượng dữ liệu, không phải đầu ra model.</p>
<select id="list"></select><video id="v" controls></video><pre id="info"></pre>
<button onclick="save('keep')">Keep</button><button onclick="save('reject')">Reject</button><button onclick="save('uncertain')">Uncertain</button><p id="message"></p>
<script>let rows=[],index=0;const list=document.getElementById('list');
function show(){let r=rows[index];document.getElementById('v').src='/media/'+r.sample_id;document.getElementById('info').textContent=JSON.stringify(r,null,2);list.value=index;}
list.onchange=()=>{index=Number(list.value);show()};
fetch('/items').then(r=>r.json()).then(x=>{rows=x;rows.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(i+1)+' '+r.sample_id;list.append(o)});if(rows.length)show()});
async function save(decision){let response=await fetch('/decision/'+rows[index].sample_id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decision})});if(!response.ok){document.getElementById('message').textContent=await response.text();return}rows[index].decision=decision;index=Math.min(index+1,rows.length-1);show()}
</script>"""
