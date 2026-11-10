"""Sinh 4 notebook Kaggle (generate / prepare / train / demo): python tools/make_notebooks.py ."""

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
# Log cài đặt của phiên (git, pip, worker): ghi đủ ra file, cell "Log setup" đưa lên W&B + Output.
LOGS = Path("/kaggle/temp/vn-av-df-logs")
LOGS.mkdir(parents=True, exist_ok=True)
TIMINGS = {}
def logged(name, command, **kw):
    tick = time.time()
    path = LOGS / f"{name}.log"
    with open(path, "a", encoding="utf-8") as log:
        log.write("$ " + " ".join(map(str, command)) + "\\n")
        log.flush()
        code = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, **kw).returncode
    TIMINGS[name] = round(TIMINGS.get(name, 0) + time.time() - tick, 1)
    if code:
        print("\\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]))
        raise RuntimeError(f"{name} lỗi (mã {code}); log đầy đủ: {path}")
if not ROOT.exists():
    logged("git-clone", ["git", "clone", REPO_URL, str(ROOT)])
# Weight và repo upstream (vài GB) ở /kaggle/temp: Output (giới hạn 20 GB) không bao giờ chứa
# chúng, kể cả khi phiên lỗi giữa chừng (bước dọn Output cuối không chạy).
for name in ("weights", "external"):
    (Path("/kaggle/temp/vn-av-df-heavy") / name).mkdir(parents=True, exist_ok=True)
    if not (ROOT / name).exists():
        (ROOT / name).symlink_to(Path("/kaggle/temp/vn-av-df-heavy") / name, target_is_directory=True)
sys.path.insert(0, str(ROOT))
logged("pip-install", [sys.executable, "-m", "pip", "install", "-e", str(ROOT) + "[data,EXTRA]"])
import settings
settings.bootstrap()
"""


def setup_cell(extra):
    return SETUP_TEMPLATE.replace("EXTRA", extra)


WANDB_CELL = """
# W&B: log quá trình (setup từng phiên, trích feature, nhãn, train, test, báo cáo) làm bằng chứng.
# Project PRIVATE. Thêm secret WANDB_API_KEY: Add-ons → Secrets. Không đẩy ảnh / video người thật.
USE_WANDB = True
WANDB_PROJECT = "vn-av-df-v005"  # v004: "vn-av-df-v004", v003: "vn-av-df-v003" (giữ nguyên, không ghi chồng)
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

SETUP_LOG = """
# Ghi cách cài phiên vào Output: commit, phần cứng, gói pip của kernel và từng worker, log cài (pip,
# worker), dữ liệu / cache / tài sản gắn vào. Run W&B của phiên (prepare / train / labels / report) tự
# đính kèm: config `setup`, bảng `setup/*`, artifact `setup-log`. Mở run nào cũng biết nó được cài thế nào.
from vn_av_df.setup_log import log_setup
cfg["setup"] = log_setup(cfg, SESSION_NAME, SETUP_FOLDER, logs=sorted(LOGS.glob("*.log")), env_root=WORKERS,
                         info=SETUP_INFO, timings=TIMINGS)
"""

prepare = nb(
    "prepare",
    [
        (
            "markdown",
            """
# v005 · Trích feature (AV-HuBERT Large, crop miệng, khung LIPINC-V2, XLS-R 5 nhóm lớp, waveform)

Mỗi part (`PART` = 1, 2, 3) chạy **ba phiên** bản gốc (+ 2 phiên cho mỗi biến thể v005: `-cpu` rồi GPU), mỗi phiên dưới 12 giờ và Output dưới 20 GB. Đặt tên notebook Kaggle đúng `SESSION_NAME` (`vn-av-df-prepare-p<PART>-<SESSION>`) để notebook train tìm được Output:
1. `SESSION = "landmarks"` (Accelerator: None, không tốn quota GPU): dlib dò mặt (part 1 v003: 8,7 giờ).
2. Sau phiên 1 của **cùng part**, chạy song song hai phiên, cả hai attach Output phiên 1 vào `OTHER_CACHES`:
   - `SESSION = "cpu"` (None): `mouth`, `lipinc`, `wave`.
   - `SESSION = "gpu"` (T4 ×2): `avhubert` (ghi thêm `avhubert/sync_profile.json` cho nhãn mismatch v004), `xlsr`.
3. v005, mỗi biến thể dữ liệu (`settings.VARIANTS`) **hai phiên** (một phiên làm cả ba encoder ≈ 22 GB / part, vượt Output 20 GB):
   - `SESSION = "<biến thể>-cpu"` (Accelerator: None, không tốn quota GPU), attach Output phiên `landmarks` cùng part: tạo video biến thể (giữ số frame, ~0,05-0,9 MB / clip) + trích `mouth` (~6 GB / part).
   - Xong thì `SESSION = "<biến thể>"` (T4 ×2, cả hai GPU), attach Output `landmarks` + `<biến thể>-cpu` cùng part: trích `avhubert`, `xlsr` trên clip đã nén (~17 GB / part).
   Các biến thể:
   - `SESSION = "c480"`, `"c360"`: nén lại mọi mẫu (480p libx264 CRF 30 + AAC 64k; 360p CRF 35 + AAC 32k), dùng khi train + test.
   - `SESSION = "pr"`: real / sham / voice qua xử lý lành tính giống đường ống generator (x264 CRF 8 → codec / co giãn → x264 CRF 18, tiếng giữ nguyên, nhãn giữ nguyên); train / validation 4 họ xử lý, test một nửa họ đã thấy + một nửa 3 họ chưa thấy (`clips/reencode.json` → `chains`). Âm khó khi train + phép thử "AI thật hay vết đường ống".
   - `SESSION = "w360"` (VP9 360p + Opus), `"c288"` (288p): chỉ mẫu test, mức không có khi train (ngắn).
   Clip nén nằm lại Output phiên `-cpu` (nhẹ); dở giữa chừng thì phiên sau (`PREVIOUS_CACHE`) không làm lại mẫu đã có.

Hết giờ giữa chừng: chạy lại cùng `SESSION`, attach Output phiên dở vào `PREVIOUS_CACHE`. Cell "Log setup" ghi cách cài phiên vào `cache/vn-av-df-data/setup/`; run W&B `prepare` của phiên đính kèm luôn (config `setup`, bảng `setup/*`, artifact `setup-log`).
""",
        ),
        ("code", setup_cell("training")),
        ("code", WANDB_CELL),
        (
            "code",
            """
PART = 1               # 1, 2, 3
SESSION = "landmarks"  # "landmarks" (CPU) → song song "cpu" (CPU), "gpu" (T4 ×2), "<biến thể>-cpu" (CPU) → "<biến thể>" (T4 ×2)
# Slug dataset Kaggle của từng part (bản có sync_syncnet.json); sửa nếu upload tên khác.
DATASETS = {1: "linhxm/vn-av-df-data-part1-trim04-fake",
            2: "linhxm/vn-av-df-data-part2-fake",
            3: "linhxm/vn-av-df-data-part3-fake"}
DATASET_PARTS = [Path(f"/kaggle/input/datasets/{DATASETS[PART]}")]
# Hết giờ giữa chừng: chạy lại cùng SESSION, attach Output phiên dở vào đây (mẫu đã có được bỏ qua).
PREVIOUS_CACHE = None  # Path("/kaggle/input/<Output phiên dở>/vn-av-df-forensics/cache/vn-av-df-data")
# Phiên "cpu" / "gpu" / "<biến thể>-cpu": Output phiên "landmarks" CỦA CÙNG PART (đường dẫn ở panel Input);
# phiên "<biến thể>" (GPU): thêm Output phiên "<biến thể>-cpu" cùng part (dùng lại clip đã nén).
OTHER_CACHES = []      # vd. [Path("/kaggle/input/notebooks/linhxm/vn-av-df-prepare-p1-landmarks/vn-av-df-forensics/cache/vn-av-df-data")]
SESSION_NAME = f"prepare-p{PART}-{SESSION}"
SMOKE = False          # thử luồng: chỉ 2 người mỗi split, không W&B (Output không dùng để train; xong thì đặt lại False)
MODELS = settings.MODELS + settings.ABLATIONS
import torch
from vn_av_df.dataset import part_selection
from vn_av_df.training.prepare import encoders_for
GPU = torch.cuda.is_available()
settings.DEVICE = "cuda" if GPU else "cpu"
cfg = settings.config()
CHOOSE = {
    "landmarks": lambda e: e == "landmarks",
    "cpu": lambda e: e != "landmarks" and cfg["encoders"][e].get("cpu_only"),
    "gpu": lambda e: not cfg["encoders"][e].get("cpu_only"),
    # Biến thể (settings.VARIANTS: nén, PR, mức chỉ test): landmark dùng lại của bản gốc; model v005 chỉ cần 3 encoder.
    # "<v>-cpu" (CPU, không tốn quota): nén clip + mouth; "<v>" (T4 ×2): avhubert + xlsr trên clip đó.
    # Một phiên làm cả 3 encoder ~2,9 MB / mẫu × ~7.700 mẫu ≈ 22 GB / part, vượt Output 20 GB (đo 09/11).
    **{f"{v}-cpu": (lambda e: e == "mouth") for v in settings.VARIANTS},
    **{v: (lambda e: e in ("avhubert", "xlsr")) for v in settings.VARIANTS},
}
ENCODERS = [e for e in encoders_for(MODELS) if CHOOSE[SESSION](e)]  # hoặc chọn tay
VARIANT = SESSION.removesuffix("-cpu")
if VARIANT in settings.VARIANTS:
    cfg["variant"] = dict(settings.VARIANTS[VARIANT])
if (SESSION == "gpu" or SESSION in settings.VARIANTS) and not GPU:
    raise RuntimeError(f'SESSION = "{SESSION}" cần Accelerator GPU (T4 ×2)')
if SESSION in settings.VARIANTS and not any((Path(c) / "clips").is_dir() for c in OTHER_CACHES):
    print(f'CẢNH BÁO: chưa attach Output phiên "{SESSION}-cpu" vào OTHER_CACHES, phiên GPU sẽ tự nén clip (tốn quota)')
if (SESSION in ("landmarks", "cpu") or SESSION.endswith("-cpu")) and GPU:
    print(f"CẢNH BÁO: phiên {SESSION} chỉ cần CPU, chọn Accelerator None để không tốn quota GPU")
if SESSION != "landmarks" and not OTHER_CACHES:
    print("CẢNH BÁO: OTHER_CACHES rỗng, worker sẽ tự dò landmark bằng dlib (chậm)")
print("Phiên", SESSION_NAME, "→ ENCODERS =", ENCODERS)
root, parts = part_selection(DATASET_PARTS)
cfg["dataset"], cfg["dataset_parts"] = str(root), parts
cfg["prepare_devices"] = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or [None]
cfg["wandb"] = WANDB
cfg["session"] = SESSION_NAME
if SMOKE:
    cfg["smoke_speakers"] = 2
    cfg["wandb"] = None
OUTPUT_CACHE = Path(cfg["cache"])  # ghi vào đây; đọc thêm từ OTHER_CACHES
cfg["run_group"] = OUTPUT_CACHE.name  # cùng group W&B với run prepare
if PREVIOUS_CACHE and not OUTPUT_CACHE.exists():
    shutil.copytree(PREVIOUS_CACHE, OUTPUT_CACHE)
cfg["cache"] = [str(OUTPUT_CACHE), *map(str, OTHER_CACHES)]
""",
        ),
        (
            "code",
            """
from vn_av_df.cli import setup
tick = time.time()
ASSETS = setup(cfg, generators=[], encoders=ENCODERS)  # chỉ tải thứ ENCODERS cần
TIMINGS["assets"] = round(time.time() - tick, 1)
print(ASSETS)
if {"landmarks", "avhubert"} & set(ENCODERS):
    # Cùng env cho cả hai stage; phiên CPU cài không cần GPU (stage landmarks chỉ dùng dlib).
    logged("worker-avhubert", [sys.executable, str(ROOT / "environments/bootstrap_worker.py"), "avhubert",
                               "--root", str(ROOT), "--env-root", str(WORKERS)] + ([] if GPU else ["--allow-cpu"]))
    cfg["encoders"]["avhubert"]["python"] = str(WORKERS / ".venv-avhubert/bin/python")
""",
        ),
        ("markdown", "## Log setup"),
        (
            "code",
            """
SETUP_FOLDER = OUTPUT_CACHE / "setup"
SETUP_INFO = {"part": PART, "session": SESSION, "encoders": ENCODERS, "datasets": [str(p) for p in DATASET_PARTS],
              "other_caches": [str(p) for p in OTHER_CACHES], "previous_cache": str(PREVIOUS_CACHE or ""),
              "assets": ASSETS}
"""
            + SETUP_LOG,
        ),
        (
            "code",
            """
from vn_av_df.training.prepare import prepare
from vn_av_data.common.runtime import prune_output
try:
    print(prepare(cfg, ENCODERS))
    if SESSION in settings.VARIANTS:  # phiên GPU xong: bỏ clip nén tự làm khỏi Output (clip của "-cpu" giữ ở đó)
        clips = sorted((OUTPUT_CACHE / "clips").glob("*.mp4"))
        for clip in clips:
            clip.unlink()
        print("Đã xoá", len(clips), "clip nén; giữ clips/reencode.json")
finally:  # Output chỉ giữ cache (kèm log setup), kể cả khi lỗi giữa chừng (phiên sau dùng PREVIOUS_CACHE)
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
# v005 · Train từng model (nhiều hàng song song trên 2 GPU), rồi test một lần + báo cáo

Mỗi phiên train (`GROUP` = `P1`…`P5`, `settings.TRAIN_SESSIONS`) là **một notebook Kaggle T4 ×2** tên `vn-av-df-train-<GROUP>`: các hàng chạy song song, **hàng đầu dùng riêng GPU0**, các hàng sau **chia GPU1** (baseline nhẹ chạy đồng thời); các bước trong hàng chạy nối tiếp. Phiên train **chỉ train** (không chấm test). Train xong hết thì chạy `GROUP = "test"`: chấm test mọi model một lần + ghép muộn + cổng + `report.md`, **một run W&B** duy nhất. Đầu mỗi phiên, cache cần được chép sang `/kaggle/temp` bằng 64 luồng (`STAGE_LOCAL`; đọc thẳng `/kaggle/input` chậm vì độ trễ từng file). Bỏ bước nào thì thêm vào `DROP`; chạy một bước lẻ: `GROUP = "<bước>"` (vd. `"M2"`). Thử luồng trước: `SMOKE = True` (vài người mỗi split, 1 epoch, không W&B).

| GROUP | Accelerator | GPU0 | GPU1 |
|---|---|---|---|
| `labels` | None (CPU) | nhãn mismatch 0 / 1 / mask + kiểm lối tắt → `sync_labels.json` (xem trước khi train) | |
| `P1` | T4 ×2 | M3-R (+ benchmark mọi model) | đồng thời: BL-M/sup → M2 → BL-M + SpeechForensics, và BL-A2 → BL-V |
| `P2` (`PREVIOUS_RUN` = P1) | T4 ×2 | M3-R chạy tiếp (xong rồi thì bỏ qua) | đồng thời: M3-R −cặp sạch ↔ nén, và M2 chạy tiếp → BL-M + SpeechForensics |
| `P3` (tuỳ chọn) | T4 ×2 | M3-R +nhánh chất lượng | M3-R không học nén |
| `P4`, `P5` (tuỳ chọn) | T4 ×2 | BL-A | M3 của v004 (trihead_v4) |
| `M1` (hoãn) | T4 ×2 | M1 | |
| `test` (`PREVIOUS_RUN` = mọi phiên train) | T4 ×2 | chấm test mọi model (bản gốc + mọi biến thể + PR), ghép muộn BL-Ens / BL-Ens-AV, cổng V / R, `report.md`; một run W&B `test` | |
| `report` | None (CPU) | dựng lại `report.md` từ kết quả test đã có (không chấm lại) | |

Cần: 3 dataset part (có `sync_syncnet.json`, chạy `training/01a_syncnet.py` ở local) + Output prepare: 9 phiên bản gốc + mỗi biến thể 2 phiên × 3 part (`c480`, `c360` bắt buộc; `pr`, `w360`, `c288` tuỳ chọn). Mọi model train thấy các biến thể có `"train": True` (c480, c360; M3-R theo cặp sạch ↔ biến thể, model khác lấy ngẫu nhiên `variant_p` = 0,5; trừ `trihead_v5_clean`, và BL-V / BL-A vì bản biến thể không có khung LIPINC / waveform) và được test thêm trên mọi biến thể (`predictions-test-<biến thể>.npz`, `metrics.json` → `variants`; PR → `pipeline`: AUC khớp đường ống, FPR theo họ đã thấy / chưa thấy). Mỗi run train trên W&B có luôn cách cài phiên (config `setup`, bảng `setup/*`, artifact `setup-log`) và `architecture`; lần test ghi `test-log.json` (số lần model dùng test). Model dừng ở `MAX_HOURS` chưa xong: chạy lại phiên đó với `PREVIOUS_RUN`, train tiếp từ `last.pt`.
""",
        ),
        ("code", setup_cell("training")),
        ("code", WANDB_CELL),
        (
            "code",
            """
GROUP = "labels"      # "labels" → "P1" → "P2" → ("P3") → ("P4") → ("P5") → "test"; hoặc một bước lẻ, vd. "M2"
DROP = []             # bước bỏ ở phiên này, vd. ["M3-abs"] (settings.TRAIN_GROUPS)
DATASET_PARTS = [Path(f"/kaggle/input/datasets/linhxm/{s}") for s in (
    "vn-av-df-data-part1-trim04-fake", "vn-av-df-data-part2-fake", "vn-av-df-data-part3-fake")]
# Output của 9 phiên prepare (3 part × landmarks / cpu / gpu), tên notebook vn-av-df-prepare-p<N>-<phiên>.
PREPARE = "/kaggle/input/notebooks/linhxm/vn-av-df-prepare-p{}-{}/vn-av-df-forensics/cache/vn-av-df-data"
CACHES = [Path(PREPARE.format(p, s)) for p in (1, 2, 3) for s in ("landmarks", "cpu", "gpu")]
# Biến thể (settings.VARIANTS, phiên prepare cùng tên của 3 part): train học biến thể "train" + test thêm trên mọi biến thể.
# Mỗi biến thể 2 phiên mỗi part: "<v>-cpu" (clip nén + mouth) và "<v>" (avhubert + xlsr).
VARIANT_CACHES = {v: [Path(PREPARE.format(p, s)) for p in (1, 2, 3) for s in (v, f"{v}-cpu")] for v in settings.VARIANTS}
RUNS = "/kaggle/input/notebooks/linhxm/vn-av-df-train-{}/vn-av-df-forensics/runs/" + settings.RUN_NAME
PREVIOUS_RUN = []     # P2: [Path(RUNS.format("p1"))]; P5: [Path(RUNS.format("p4"))]  (slug Kaggle viết thường)
                      # test / report: [Path(RUNS.format(g)) for g in ("p1", "p2")] (+ phiên khác đã chạy, vd. "p3")
SEEDS = [42]          # lượt đầu một seed; chốt thiết kế rồi chạy [42, 43, 44]
MAX_HOURS = 11.5      # tính từ đầu phiên (SESSION_START); dừng sạch (lưu last.pt) trước giới hạn 12 giờ của
                      # Kaggle, chừa ~30 phút cho phần sau train (W&B, Output)
STAGE_LOCAL = True    # chép cache cần sang /kaggle/temp trước khi train / test (nhanh hơn nhiều lần; thiếu chỗ thì tự bỏ)
STAGE_MAX_GB = 90     # trần tổng GB chép: 10/11 phiên GPU ghi ~95 GB rồi lỗi "Read-only file system" dù ổ báo còn chỗ;
                      # bản gốc chép đủ trước, biến thể chia đều phần còn lại, mẫu không chép đọc thẳng /kaggle/input
SMOKE = False         # thử luồng: 2 người mỗi split, 1 epoch, không W&B (xong thì đặt lại False)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # trước lần dùng GPU đầu
import torch
from vn_av_df.dataset import part_selection
for c in CACHES + [c for v, cs in VARIANT_CACHES.items() if settings.VARIANTS[v].get("train") for c in cs]:
    assert c.is_dir(), f"Sai đường dẫn: {c}"
# Biến thể chỉ để đo (PR, w360, c288): thiếu Output phiên prepare thì bỏ, chỉ mất bảng / cổng tương ứng.
for v in [v for v in VARIANT_CACHES if not settings.VARIANTS[v].get("train")]:
    missing = [c for c in VARIANT_CACHES[v] if not c.is_dir()]
    if missing:
        print(f"CẢNH BÁO: thiếu cache biến thể {v} ({len(missing)} part), bỏ biến thể này:", missing[0])
        VARIANT_CACHES.pop(v)
settings.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if GROUP not in ("labels", "report", "test", *settings.TRAIN_SESSIONS, *settings.TRAIN_GROUPS):
    raise ValueError(f"GROUP không có: {GROUP}")
if GROUP not in ("labels", "report") and not torch.cuda.is_available():
    raise RuntimeError(f"GROUP {GROUP} cần Accelerator GPU")
cfg = settings.config()
cfg["deadline"] = SESSION_START + MAX_HOURS * 3600
if SMOKE:
    cfg["smoke_speakers"] = 2
    cfg["training"]["max_epochs"] = 1  # trần trên mọi công thức gốc (BL-M 500, BL-Msup / BL-V / BL-A2 100 epoch)
    WANDB = None
# BL-A trọng số lớp: "balanced" = 1 − tỉ lệ lớp trong train, cùng nguyên tắc với gốc; "paper" =
# đúng 0,1 / 0,9 (cân bằng cho ASVspoof ~90 % giả). Đổi thì BL-A train lại từ đầu.
cfg["recipe_overrides"] = {"ssl_aasist": {"class_weight": "balanced"}}
root, parts = part_selection(DATASET_PARTS)
cfg["dataset"], cfg["dataset_parts"] = str(root), parts
cfg["cache"] = [str(p) for p in CACHES]
cfg["variants"] = {v: [str(p) for p in cs] for v, cs in VARIANT_CACHES.items()}
# Đội hình đầy đủ (settings.TEST_MODELS): để in model nào chưa xong; mỗi bước chỉ train / test model của nó.
cfg["models"], cfg["seeds"] = list(settings.TEST_MODELS), list(SEEDS)
cfg["wandb"] = WANDB
SESSION_NAME = f"train-{GROUP}-{time.strftime('%Y%m%d-%H%M')}"
cfg["session"], cfg["train_group"] = SESSION_NAME, GROUP
from vn_av_df.training.train import merge_runs, missing_runs
from vn_av_df.setup_log import copy_tree
previous = [PREVIOUS_RUN] if isinstance(PREVIOUS_RUN, (str, Path)) else list(PREVIOUS_RUN or [])
for p in previous:
    assert Path(p).is_dir(), f"Sai đường dẫn: {p}"
if previous:
    print("Gộp từ phiên trước:", merge_runs(cfg["runs"], previous))
# Log setup của 9 phiên prepare đi cùng kết quả train (runs/<RUN>/setup/prepare-…).
print("Log setup prepare:", sum((copy_tree(c / "setup", Path(cfg["runs"]) / "setup") for c in CACHES), []))
if GROUP in ("labels", "report", "test"):
    STEPS = []
elif GROUP in settings.TRAIN_SESSIONS:
    STEPS = [[s for s in lane if s not in DROP] for lane in settings.TRAIN_SESSIONS[GROUP]]
else:
    STEPS = [[GROUP]]
# Mỗi hàng một GPU: model của các bước trong hàng, chạy nối tiếp; xong model nào test ngay model đó.
LANES = [[m for s in lane for group in settings.TRAIN_GROUPS[s]["lanes"] for m in group] for lane in STEPS]
LANES = [lane for lane in LANES if lane]
MODELS = [m for lane in LANES for m in lane]  # train ở phiên này
RUN_BENCHMARK = any(settings.TRAIN_GROUPS[s].get("benchmark") for lane in STEPS for s in lane)
cfg["test_after_train"] = False  # v005: test chấm một lần ở GROUP "test"
print("Phiên", GROUP, "| bước:", STEPS, "| model theo GPU:", LANES, "| chưa train xong:", missing_runs(cfg))
# Chép cache model của phiên cần (train + validation, bản gốc + biến thể) sang ổ local bằng 64 luồng: đọc
# /kaggle/input bị chặn bởi độ trễ từng file (đo 08/11: 1 luồng 10 MB/s, 64 luồng 438 MB/s), M3 v004 mất
# 75-96 phút / epoch chỉ vì đọc. Thiếu chỗ thì tự bỏ qua (đọc thẳng như cũ).
if STAGE_LOCAL and (MODELS or GROUP == "test"):
    from vn_av_df.dataset import training_dataset
    from vn_av_df.training.prepare import encoders_for
    from vn_av_df.training.stage import stage_cfg
    tick = time.time()
    _rows, _ = training_dataset(cfg, verify_media=False, labels=False)
    _splits = ("validation", "test") if GROUP == "test" else ("train", "validation")
    print("Ổ /kaggle/temp trước khi chép: còn", round(shutil.disk_usage("/kaggle/temp").free / 1e9, 1), "GB")
    print("Chép sang /kaggle/temp:", stage_cfg(cfg, encoders_for(MODELS or cfg["models"]), _rows, "/kaggle/temp/cache",
                                               splits=_splits, budget_gb=STAGE_MAX_GB))
    print("Ổ /kaggle/temp sau khi chép: còn", round(shutil.disk_usage("/kaggle/temp").free / 1e9, 1), "GB")
    TIMINGS["stage"] = round(time.time() - tick, 1)
""",
        ),
        (
            "code",
            """
from vn_av_df.cli import model_assets
tick = time.time()
ASSETS = model_assets(cfg) if MODELS or RUN_BENCHMARK else {}  # XLS-R gốc (BL-A) + checkpoint AVH-Align (BL-M, M2, M3)
TIMINGS["assets"] = round(time.time() - tick, 1)
print(ASSETS)
""",
        ),
        ("markdown", "## Log setup"),
        (
            "code",
            """
SETUP_FOLDER = Path(cfg["runs"]) / "setup"
SETUP_INFO = {"group": GROUP, "steps": STEPS, "drop": DROP,
              "titles": {s: settings.TRAIN_GROUPS[s]["title"] for lane in STEPS for s in lane},
              "lanes": LANES, "benchmark": RUN_BENCHMARK, "test_models": cfg["models"], "seeds": SEEDS,
              "previous_run": [str(p) for p in previous], "datasets": [str(p) for p in DATASET_PARTS],
              "caches": [str(p) for p in CACHES], "variant_caches": cfg["variants"], "assets": ASSETS, "max_hours": MAX_HOURS,
              "recipe_overrides": cfg["recipe_overrides"]}
"""
            + SETUP_LOG,
        ),
        (
            "code",
            """
if GROUP == "labels":
    from vn_av_df.training.sync_labels import report
    labels = report(cfg)
    print(json.dumps({k: v for k, v in labels.items() if k != "samples"}, indent=1, ensure_ascii=False)[:20000])
    print("→ gửi runs/" + settings.RUN_NAME + "/sync_labels.json để xem trước khi train")
if RUN_BENCHMARK:
    from vn_av_df.training.benchmark import benchmark
    print(json.dumps(benchmark(cfg, cfg["models"], steps=10), indent=2))
if MODELS and len(LANES) > 1 and torch.cuda.is_available():
    from vn_av_df.training.lanes import run_lanes  # hàng đầu GPU0 riêng, hàng sau chia GPU1
    run_lanes(cfg, LANES)
elif MODELS:
    from vn_av_df.training.train import train_all
    print(json.dumps(train_all({**cfg, "models": MODELS}), indent=2, ensure_ascii=False))
""",
        ),
        (
            "code",
            """
# Phiên train chỉ train; GROUP "test" chấm test mọi model một lần (một run W&B cho test + báo cáo).
from vn_av_df.training.train import run_state
runs = Path(cfg["runs"])
if MODELS:
    done = [m for m in MODELS if all(run_state(runs / f"{m}_seed{s}")[0] for s in SEEDS)]
    print("Train xong:", done, "| còn dở (chạy lại phiên với PREVIOUS_RUN):", [m for m in MODELS if m not in done])
if GROUP == "test":
    from vn_av_df import tracking
    from vn_av_df.training.evaluate import evaluate_all
    from vn_av_df.training.report import build_report
    print("Chưa train xong (không có trong test):", missing_runs(cfg))
    TEST_RUN = tracking.open_run(cfg, "test", f"{runs.name}-test", group=runs.name, folder=runs, with_setup=True)
    print(json.dumps(evaluate_all(cfg, run=TEST_RUN), indent=2))
    report = build_report(cfg, run=TEST_RUN)
    TEST_RUN.finish()
    print(Path(report["report"]).read_text(encoding="utf-8"))
if GROUP == "report":
    from vn_av_df.training.report import build_report
    report = build_report(cfg)  # đọc kết quả test đã có (GROUP "test"), không chấm lại
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
