"""Sinh 4 notebook Kaggle của v003 (generate / prepare / train / demo): python tools/make_notebooks.py ."""

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])


def nb(name, cells):
    out = []
    for i, (kind, text) in enumerate(cells):
        lines = text.strip("\n").splitlines(keepends=True)
        cell = {"cell_type": kind, "id": f"{name}-{i:02d}", "metadata": {}, "source": lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_TEMPLATE = """
from pathlib import Path
import subprocess, sys, os, time, json, shutil
SESSION_START = time.time()
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
REPO_URL = "https://github.com/linhxm/vn-av-df-forensics.git"  # Sửa thành repository của bạn.
ROOT = Path("/kaggle/working/vn-av-df-forensics")
WORKERS = Path("/kaggle/temp/vn-av-df-workers")  # Môi trường worker: ngoài Output 20 GB.
if not ROOT.exists():
    subprocess.run(["git", "clone", REPO_URL, str(ROOT)], check=True)
sys.path.insert(0, str(ROOT))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT / "data_pipeline"),
                "-e", str(ROOT) + "[EXTRA]"], check=True)
import settings
settings.bootstrap()
"""


def setup_cell(extra):
    return SETUP_TEMPLATE.replace("EXTRA", extra)


WANDB_CELL = """
# W&B: log quá trình train (trích feature, train, test, báo cáo) làm bằng chứng. Project PRIVATE.
# Thêm secret WANDB_API_KEY: Add-ons → Secrets. Không đẩy ảnh / video người thật.
USE_WANDB = True
WANDB_PROJECT = "vn-av-df-v003"
WANDB_ENTITY = "vn-av-df-capstone"  # team trong org vn-av-df-capstone-org; key tạo trong org đó
if USE_WANDB:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    os.environ["WANDB_DIR"] = "/kaggle/temp"  # file tạm W&B ở ngoài Output
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
WANDB = {"project": WANDB_PROJECT, "entity": WANDB_ENTITY} if USE_WANDB else None
"""

ENV_HELPER = '''
def ensure_env(name):
    """Cài (hoặc dùng lại) môi trường worker và trả python của nó."""
    try:
        out = subprocess.run([sys.executable, str(ROOT / "environments/worker_envs.py"), name,
                              "--root", str(ROOT), "--env-root", str(WORKERS)],
                             check=True, capture_output=True, text=True)
        print(out.stdout[-2000:])
    except subprocess.CalledProcessError as exc:
        log = WORKERS / ".kaggle-tools" / name / "setup.log"
        if log.exists():
            print("\\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]))
        raise RuntimeError(f"Cài môi trường {name} lỗi; xem log ở trên") from exc
    return str(WORKERS / f".venv-{name}/bin/python")
'''

generate = nb(
    "generate",
    [
        (
            "markdown",
            """
# v003 · Sinh fake 6 cell trên Kaggle

Mỗi phiên làm **một phần** của một part sạch:
1. **Phiên giọng** (`PHASE = "voice"`): ① real, ② sham, ③ giọng AI (track A/B, test thêm track C) và track B cho ⑥. Cần GPU cho TTS. Chạy **trước**.
2. **Phiên lip-sync** (`PHASE = "lipsync"`, `GENERATORS = [...]`): ④⑤⑥, mỗi phiên vài generator.
   Phiên sau attach Output phiên trước vào `PREVIOUS_OUTPUT` (danh sách), job đã xong được bỏ qua.
3. Phiên cuối (đủ mọi job): cell 6 tự finalize → `datasets/vn-av-df-data/<part>/manifest.jsonl`.

| | |
|---|---|
| Đầu vào | Part sạch v003 (data_pipeline/exports/…/vn-av-df-data-partN) upload thành Kaggle dataset |
| Đầu ra | `vn-av-df-forensics/datasets/vn-av-df-data/<part>/` + `generation/plans/<part>.json` |
| Lưu ý | Môi trường worker generator / TTS **chưa chạy thử trên Kaggle**: lỗi cài thì xem log in ra, sửa công thức trong `environments/worker_envs.py` |
| IP-LAP | Weight chỉ có trên OneDrive: tải tay 2 file `.pth`, upload thành dataset, đặt `IPLAP_CHECKPOINTS` |
""",
        ),
        ("markdown", "## 1. Cài đặt"),
        ("code", setup_cell("generation")),
        ("markdown", "## 2. Chọn part, phiên, generator"),
        (
            "code",
            """
PART = 1
CLEAN_PART = Path("/kaggle/input/datasets/YOUR_USER/vn-av-df-clean-part1")  # thư mục có manifest.jsonl
PHASE = "voice"          # "voice" trước, rồi "lipsync"
GENERATORS = ["wav2lip_gan", "musetalk_1_5"]  # phiên lip-sync: chia nhóm theo VRAM / thời gian
PREVIOUS_OUTPUT = []     # [Path("/kaggle/input/<phiên trước>/vn-av-df-forensics"), ...]
BATCH = None             # (k, n): chỉ chạy đợt k trong n đợt clip, vd. (1, 4)
IPLAP_CHECKPOINTS = None # Path("/kaggle/input/datasets/YOUR_USER/iplap-checkpoints")
TIME_BUDGET_HOURS = 11
SEED = 42
""",
        ),
        ("markdown", "## 3. Plan và gộp phiên trước"),
        (
            "code",
            """
from vn_av_data.common.runtime import find_part
import torch
settings.PART, settings.GENERATION_NAME = PART, f"vn-av-df-data-part{PART}"
settings.CLEAN_DATASET = find_part(CLEAN_PART, "manifest.jsonl")
settings.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = settings.config()
cfg["seed"] = SEED
cfg["generation"]["iplap_checkpoints"] = str(IPLAP_CHECKPOINTS) if IPLAP_CHECKPOINTS else None
from vn_av_df.generation.generate import merge_previous
from vn_av_df.generation.plan import make_plan
if PREVIOUS_OUTPUT:
    print(merge_previous(cfg, PREVIOUS_OUTPUT))
if not Path(cfg["plan"]).exists():
    print(make_plan(cfg["clean_dataset"], cfg["plan"], cfg["generation"], SEED))
plan = json.loads(Path(cfg["plan"]).read_text(encoding="utf-8"))
print({k: len([j for j in plan["jobs"] if j["type"] == k]) for k in ("voice", "lipsync")})
""",
        ),
        ("markdown", "## 4. Cài worker của phiên"),
        (
            "code",
            ENV_HELPER
            + """
from vn_av_df.generation.voice.engines import envs_for
from vn_av_df.generation.lipsync import ENVS as GEN_ENVS
if PHASE == "voice":
    methods = {j["voice"] for j in plan["jobs"] if j["type"] == "voice"}
    methods |= {j["inpaint"] for j in plan["jobs"] if j["type"] == "voice" and j.get("inpaint")}
    for env in envs_for(methods):
        cfg["envs"][env] = ensure_env(env)
else:
    from vn_av_df.cli import setup
    print(setup(cfg, GENERATORS))  # clone source + tải weight (trước khi cài môi trường)
    if "musetalk_1_5" in GENERATORS:
        subprocess.run([sys.executable, str(ROOT / "environments/bootstrap_worker.py"), "musetalk",
                        "--root", str(ROOT), "--env-root", str(WORKERS)], check=True)
        cfg["musetalk"]["python"] = str(WORKERS / ".venv-musetalk/bin/python")
    for name in GENERATORS:
        if name in GEN_ENVS:
            cfg["envs"][GEN_ENVS[name]] = ensure_env(GEN_ENVS[name])
print(cfg["envs"])
""",
        ),
        ("markdown", "## 5. Sinh"),
        (
            "code",
            """
import traceback
from vn_av_df.generation.generate import run_generation
cfg["generation_gpus"] = list(range(torch.cuda.device_count())) or [None]
cfg["generation_deadline"] = SESSION_START + TIME_BUDGET_HOURS * 3600
if PHASE == "lipsync":
    cfg["session_generators"] = GENERATORS
cfg["session_batch"] = BATCH
error = None
try:
    print(run_generation(cfg, PHASE))
except Exception as exc:
    error = exc
    traceback.print_exc()
""",
        ),
        ("markdown", "## 6. Finalize (khi đủ mọi job) và dọn Output"),
        (
            "code",
            """
from vn_av_df.generation.generate import finalize
out = Path(cfg["generated_dataset"])
done = {p.stem for p in (out / "records").glob("*.json")}
left = [j for j in plan["jobs"] if j["key"] not in done]
print("Còn", len(left), "job:", dict(__import__("collections").Counter(j.get("generator", "voice") for j in left)))
if not left:
    print(finalize(cfg))
from vn_av_data.common.runtime import prune_output
keep = [str(out.relative_to(ROOT)), str(Path(cfg["plan"]).relative_to(ROOT))]
print("Đã xoá khỏi Output:", len(prune_output(ROOT, keep)), "mục")
if error:
    raise error
""",
        ),
    ],
)

prepare = nb(
    "prepare",
    [
        (
            "markdown",
            """
# v003 · Trích feature (AV-HuBERT Large, crop miệng, khung LIPINC-V2, XLS-R, waveform)

Đầu vào: dataset đã finalize của các part (`datasets/vn-av-df-data/<part>`). Đầu ra: `cache/vn-av-df-data/{avhubert,mouth,lipinc,xlsr,wave}` (chỉ encoder mà `MODELS` cần).
Hết giờ phiên thì Save Version, phiên sau attach `PREVIOUS_CACHE` để trích tiếp (cache đã có được bỏ qua).
Cache một part cỡ vài chục GB: nếu vượt giới hạn Output một phiên, mỗi phiên chỉ trích 1-2 encoder (`ENCODERS`) rồi Save Version thành **dataset riêng**. Phiên trích `mouth` / `lipinc` cần landmark của `avhubert`: attach dataset đó vào `OTHER_CACHES`. Notebook train nhận danh sách các dataset cache.
""",
        ),
        ("code", setup_cell("training")),
        ("code", WANDB_CELL),
        (
            "code",
            """
DATASET_PARTS = [Path("/kaggle/input/datasets/YOUR_USER/vn-av-df-data-part1")]
PREVIOUS_CACHE = None  # Path("/kaggle/input/<phiên trước>/vn-av-df-forensics/cache/vn-av-df-data")
OTHER_CACHES = []      # dataset cache khác chỉ để đọc, vd. [Path(".../vn-av-df-features-part1-avhubert")]
MODELS = settings.MODELS + settings.ABLATIONS
import torch
from vn_av_df.dataset import part_selection
from vn_av_df.training.prepare import encoders_for
ENCODERS = encoders_for(MODELS)  # hoặc chọn tay, ví dụ ["avhubert", "mouth"]
settings.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = settings.config()
root, parts = part_selection(DATASET_PARTS)
cfg["dataset"], cfg["dataset_parts"] = str(root), parts
cfg["prepare_devices"] = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or [None]
cfg["wandb"] = WANDB
OUTPUT_CACHE = Path(cfg["cache"])  # ghi vào đây; đọc thêm từ OTHER_CACHES
if PREVIOUS_CACHE and not OUTPUT_CACHE.exists():
    shutil.copytree(PREVIOUS_CACHE, OUTPUT_CACHE)
cfg["cache"] = [str(OUTPUT_CACHE), *map(str, OTHER_CACHES)]
""",
        ),
        (
            "code",
            """
from vn_av_df.cli import setup
print(setup(cfg, generators=[], encoders=True))  # AV-HuBERT Large + XLS-R + checkpoint AVH-Align
if "avhubert" in ENCODERS:
    subprocess.run([sys.executable, str(ROOT / "environments/bootstrap_worker.py"), "avhubert",
                    "--root", str(ROOT), "--env-root", str(WORKERS)], check=True)
    cfg["encoders"]["avhubert"]["python"] = str(WORKERS / ".venv-avhubert/bin/python")
""",
        ),
        (
            "code",
            """
from vn_av_df.training.prepare import prepare
print(prepare(cfg, ENCODERS))
from vn_av_data.common.runtime import prune_output
print("Đã xoá khỏi Output:", len(prune_output(ROOT, [str(OUTPUT_CACHE.relative_to(ROOT))])), "mục")
""",
        ),
    ],
)

train = nb(
    "train",
    [
        (
            "markdown",
            """
# v003 · Train + test một lần + báo cáo

Cần: dataset đã finalize + cache feature (notebook prepare). Model: `settings.MODELS` (4 baseline + M1, M2) + `settings.ABLATIONS` (3 ablation M1); BL-Ens tính ở bước test.
Đầu ra: `runs/<RUN_NAME>/` (checkpoint, history, metrics, report.md, gates.json, test-lock.json, provenance.json).
Hết giờ phiên giữa chừng: Save Version, phiên sau attach Output vào `PREVIOUS_RUN`; model xong được bỏ qua, model đang dở train tiếp từ epoch sau `last.pt`.
Chạy song song nhiều notebook: mỗi notebook một nhóm `MODELS`, `RUN_TEST = False`. Notebook cuối đặt `PREVIOUS_RUN` = danh sách Output các notebook đó, `MODELS` đủ: model đã xong được gộp và bỏ qua, test + report chỉ chạy khi đủ mọi model.
`RUN_BENCHMARK = True`: đo thời gian / bộ nhớ từng model trên cache thật trước (ghi `benchmark.json`), để chia phiên.
""",
        ),
        ("code", setup_cell("training")),
        ("code", WANDB_CELL),
        (
            "code",
            """
DATASET_PARTS = [Path("/kaggle/input/datasets/YOUR_USER/vn-av-df-data-part1")]
# Một hoặc nhiều dataset cache (gộp lại phải đủ avhubert/ mouth/ lipinc/ xlsr/ wave/)
CACHES = [Path("/kaggle/input/datasets/YOUR_USER/vn-av-df-features-part1")]
PREVIOUS_RUN = None   # Path(".../runs/train_part1_v003") hoặc danh sách nhiều Output để gộp
MODELS = settings.MODELS + settings.ABLATIONS
SEEDS = [42]          # đánh giá part 1: một seed; chốt thiết kế rồi chạy [42, 43, 44]
RUN_TEST = True
RUN_BENCHMARK = False
import torch
from vn_av_df.dataset import part_selection
settings.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = settings.config()
root, parts = part_selection(DATASET_PARTS)
cfg["dataset"], cfg["dataset_parts"] = str(root), parts
cfg["cache"] = [str(p) for p in CACHES]
cfg["models"], cfg["seeds"] = list(MODELS), list(SEEDS)
cfg["wandb"] = WANDB
from vn_av_df.training.train import merge_runs, missing_runs
previous = [PREVIOUS_RUN] if isinstance(PREVIOUS_RUN, (str, Path)) else list(PREVIOUS_RUN or [])
if previous:
    print("Gộp từ phiên trước:", merge_runs(cfg["runs"], previous))
""",
        ),
        (
            "code",
            """
from vn_av_df.cli import model_assets
print(model_assets(cfg))  # XLS-R gốc (BL-A fine-tune) + checkpoint AVH-Align công bố (BL-M, M2)
if RUN_BENCHMARK:
    from vn_av_df.training.benchmark import benchmark
    print(json.dumps(benchmark(cfg, MODELS, steps=10), indent=2))
from vn_av_df.training.train import train_all
print(json.dumps(train_all(cfg), indent=2, ensure_ascii=False))
""",
        ),
        (
            "code",
            """
missing = missing_runs(cfg)
if RUN_TEST and missing:
    print("Chưa đủ model, chưa chạy test (test chỉ chạy một lần, khi đủ):", missing)
elif RUN_TEST:
    from vn_av_df.training.evaluate import evaluate_all
    from vn_av_df.training.report import build_report
    print(json.dumps(evaluate_all(cfg), indent=2))
    report = build_report(cfg)
    print(Path(report["report"]).read_text(encoding="utf-8"))
from vn_av_data.common.runtime import prune_output
print("Đã xoá khỏi Output:", len(prune_output(ROOT, [str(Path(cfg["runs"]).relative_to(ROOT))])), "mục")
""",
        ),
    ],
)

demo = nb(
    "demo",
    [
        (
            "markdown",
            """
# v003 · Demo: tạo fake từ video real + phân tích video

Chạy ở **phiên tương tác** (GPU, bật Internet), rồi mở link `gradio.live` in ra ở cell cuối.

| Tab | Làm gì |
|---|---|
| 1. Tạo fake | Tải video real ≤ 60 s → chọn ô ③④⑤⑥, full/partial, generator, hệ giọng → video fake + nhãn thật |
| 2. Phân tích | Tải video (hoặc dùng video vừa tạo) → sửa độ lệch tiếng-hình → điểm 4 tác vụ của từng model + timeline, vẽ chồng nhãn thật |

Cần: output train (`runs/<RUN_NAME>/`, có `best.pt` và tốt nhất là `metrics.json` để có ngưỡng). Chưa có thì để
`RUN_DIR = None`: demo chỉ chạy `speechforensics` (không cần train).
Video người thật: chỉ dùng video của mình hoặc đã xin phép; link gradio.live là công khai trong 72 giờ, tắt khi xong.
""",
        ),
        ("code", setup_cell("generation,training,demo")),
        (
            "code",
            """
RUN_DIR = None        # Path("/kaggle/input/<output train>/vn-av-df-forensics/runs/train_part1_v003")
MODELS = None         # None = mọi model có trong RUN_DIR; hoặc ["trihead_v2", "avh_align_plus", "lipinc_v2"]
GENERATORS = ["wav2lip_gan", "latentsync_1_5"]  # generator cài sẵn cho tab "Tạo fake"
VOICES = ["f5_vi"]                               # hệ giọng cài sẵn
import torch
settings.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = settings.config()
""",
        ),
        (
            "code",
            ENV_HELPER
            + """
# Model nhận dạng mặt / SyncNet của data pipeline (đo + sửa độ lệch tiếng-hình)
subprocess.run([sys.executable, str(ROOT / "data_pipeline/steps/00_setup.py")], check=True)
from vn_av_df.cli import setup
print(setup(cfg, GENERATORS, encoders=True))  # source + weight generator, AV-HuBERT Large, XLS-R
subprocess.run([sys.executable, str(ROOT / "environments/bootstrap_worker.py"), "avhubert",
                "--root", str(ROOT), "--env-root", str(WORKERS)], check=True)
cfg["encoders"]["avhubert"]["python"] = str(WORKERS / ".venv-avhubert/bin/python")
from vn_av_df.generation.lipsync import ENVS as GEN_ENVS
from vn_av_df.generation.voice.engines import envs_for
if "musetalk_1_5" in GENERATORS:
    subprocess.run([sys.executable, str(ROOT / "environments/bootstrap_worker.py"), "musetalk",
                    "--root", str(ROOT), "--env-root", str(WORKERS)], check=True)
    cfg["musetalk"]["python"] = str(WORKERS / ".venv-musetalk/bin/python")
for name in GENERATORS:
    if name in GEN_ENVS:
        cfg["envs"][GEN_ENVS[name]] = ensure_env(GEN_ENVS[name])
for env in envs_for(set(VOICES)):
    cfg["envs"][env] = ensure_env(env)
print(cfg["envs"])
""",
        ),
        (
            "code",
            """
from vn_av_df.demo.app import build_app
from vn_av_df.demo.detect import Detector, training_free_run
run_dir = RUN_DIR or training_free_run(Path("/kaggle/temp/demo-run"))
detector = Detector(cfg, run_dir, MODELS)
print("Model:", list(detector.models))
app = build_app(cfg, detector, GENERATORS, VOICES, work=Path("/kaggle/temp/demo"))
app.launch(share=True, debug=True)
""",
        ),
    ],
)

for name, book in (("generate", generate), ("prepare", prepare), ("train", train), ("demo", demo)):
    (ROOT / "notebooks" / f"{name}.ipynb").write_text(
        json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
print("ok")
