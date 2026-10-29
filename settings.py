"""Cấu hình mặc định cho các file .py ở local (generation/, training/).

Data/review: chọn PART trong data_settings.py. Trên Kaggle: chỉnh các cell cấu hình trong
notebook, không cần sửa file này rồi upload lại. config() lấy giá trị hiện tại; trong notebook
chỉ gọi nó sau khi chọn đường dẫn/model. RUN_NAME là folder kết quả, không phải tên dataset.
"""

import os
import sys
from pathlib import Path

from data_settings import DATASET_NAME, PART, part_name

ROOT = Path(__file__).resolve().parent
CLEAN_DATASET = ROOT / "data_pipeline/exports" / DATASET_NAME / part_name()
GENERATION_NAME = part_name()
DATASET = ROOT / "datasets" / DATASET_NAME
# "all" = mọi part đã tải vào DATASET; hoặc ["vn-av-df-data-part1", ...].
DATASET_PARTS = [part_name()]
RUN_NAME = "train_part1_v003"  # Output: runs/<RUN_NAME>/<model>_seed<seed>/.
DEVICE = "cuda"  # Kaggle GPU. Use "cpu" for a local CPU-only environment.

# Generator và hệ giọng: nhóm đầu dùng cho train/validation/test, nhóm sau chỉ ở test.
TRAIN_GENERATORS = ["wav2lip_gan", "musetalk_1_5", "iplap", "videoretalking"]
TEST_ONLY_GENERATORS = ["latentsync_1_5", "diff2lip"]
TRAIN_VOICES = ["vieneu", "f5_vi", "omnivoice", "seedvc"]
TEST_ONLY_VOICES = ["voxcpm2", "vixtts"]
# Đội hình chốt 23/10 (docs/v003_model.md §4-§5): 4 baseline cài lại đúng code gốc (BL-V, BL-A,
# BL-M chỉ real, BL-M/sup) + M1 + M2; BL-Ens = max(BL-V, BL-A) tính ở evaluate, không train.
MODELS = [
    "lipinc_v2",
    "ssl_aasist",
    "avh_align",
    "avh_align_sup",
    "trihead_v2",
    "avh_align_plus",
]
# Ablation của M1 (bảng phụ): bỏ nhánh sync, bỏ nhánh trace, bỏ tham chiếu người.
ABLATIONS = [
    "trihead_v2_no_sync",
    "trihead_v2_no_trace",
    "trihead_v2_no_ref",
]
SEEDS = [42]  # giai đoạn đánh giá part 1: một seed; chốt thiết kế rồi mới chạy [42, 43, 44]
# Python của từng môi trường worker (environments/worker_envs.py in ra đường dẫn sau khi cài).
ENVS = {}
AVH_PYTHON = os.environ.get(
    "VN_AVH_PYTHON",
    str(ROOT / ".venv-avhubert" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
)
MUSETALK_PYTHON = os.environ.get(
    "VN_MUSETALK_PYTHON",
    str(ROOT / ".venv-musetalk" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
)


def config():
    """Cấu hình dùng chung cho Run File và notebook; không hardcode thuật toán trong notebook."""
    train = {"generators": TRAIN_GENERATORS, "voices": TRAIN_VOICES}
    test = {
        "generators": TRAIN_GENERATORS + TEST_ONLY_GENERATORS,
        "voices": TRAIN_VOICES + TEST_ONLY_VOICES,
    }
    cfg = {
        "seed": 42,
        "seeds": SEEDS,
        "device": DEVICE,
        "root": str(ROOT),
        "clean_dataset": str(CLEAN_DATASET),
        "generated_dataset": str(DATASET / GENERATION_NAME),
        "data_part": PART,
        "dataset": str(DATASET),
        "dataset_parts": DATASET_PARTS,
        "plan": str(ROOT / "generation/plans" / f"{GENERATION_NAME}.json"),
        "cache": str(ROOT / "cache" / DATASET_NAME),
        "runs": str(ROOT / "runs" / RUN_NAME),
        "envs": dict(ENVS),
        "models": MODELS + ABLATIONS,
        "generation": {
            "generators": {
                "train": train["generators"],
                "validation": train["generators"],
                "test": test["generators"],
            },
            "voices": {
                "train": train["voices"],
                "validation": train["voices"],
                "test": test["voices"],
            },
            "inpaint": {"test": "f5_inpaint"},  # track C (sửa trực tiếp) chỉ ở test
            # Mỗi clip chỉ chạy một nửa số generator (rút cân bằng, xoay ô); giọng giữ nguyên.
            # 20 clip / người × một nửa generator: cùng GPU với 10 clip × mọi generator nhưng gấp
            # đôi số clip real làm nền (docs/v003_generation.md §4).
            "generators_per_clip": {"train": 2, "validation": 2, "test": 3},
            "partial_range_s": [0.3, 3.0],  # gồm cả đoạn sửa 1 từ (LAV-DF: trung bình 0,65 s)
            "edge_margin_s": 0.5,
            # Tiếng donor (② sham, ⑤) căn lặng đầu / cuối như clip gốc, chọn donor vừa nhất trong
            # mọi donor dài ≥ clip gốc - shorter_s (part 1 cắt cứng: AUC khoảng lặng cuối 0,08;
            # mô phỏng trên part 1: 0,55, không clip nào thiếu donor). Bỏ khoá = cách cũ.
            "donor_fit": {"shorter_s": 1.0, "min_pause_s": 0.02},
            "blend_probability": 0.5,
            "blend_frames": [2, 6],
            "max_side": 1280,  # giữ nguyên 720p (1280×720) khi sinh
            "crf": 18,
            "asr_model": "vinai/PhoWhisper-small",
            "voice_models": str(ROOT / "weights/voice"),
            "iplap_checkpoints": None,  # thư mục có 2 file .pth của IP-LAP (tải tay từ OneDrive)
            "latentsync_config": "configs/unet/stage2.yaml",
            "references": 3,  # số clip real tham chiếu mỗi người (ref-norm, TriHead-Ref)
        },
        "wav2lip": {
            "repo": str(ROOT / "external/Wav2Lip"),
            "checkpoint": str(ROOT / "weights/wav2lip/Wav2Lip-SD-GAN.pt"),
            "device": DEVICE,
            "python": sys.executable,
        },
        "musetalk": {
            "repo": str(ROOT / "external/MuseTalk"),
            "python": MUSETALK_PYTHON,
            "device": DEVICE,
            "batch_size": 4,
            "seed": 42,
        },
        "training": {
            "epochs": 30,
            "patience": 5,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "hidden": 128,
            "dropout": 0.1,
            "batch_size": 16,
            "eval_batch_size": 8,
            "workers": 2,
            # Dịch tiếng so với hình k frame ∈ [−1, +2] khi train (dương = tiếng trễ); xem
            # docs/v003_data.md §4.3.4 (BT.1359, sai số đo, điểm 0 generator).
            "audio_lag_aug": [-1, 2],
            # Mỗi người 15-20 clip: lấy mẫu train theo 1 / số clip của người đó để mỗi người đóng
            # góp như nhau (docs/v003_model.md §7).
            "speaker_balance": True,
        },
        "bootstrap_replicates": 1000,
    }
    cfg["encoders"] = {
        "avhubert": {
            "kind": "avhubert",
            "repo": str(ROOT / "external/av_hubert"),
            # AVH-Align dùng AV-HuBERT Large (VoxCeleb2 + LRS3, fine-tune VSR 433 h).
            # Đúng checkpoint AVH-Align dùng để trích đặc trưng (README + deepfake_feature_extraction.py
            # ở commit 55401de): self-training, khác large_vox_433h.
            "checkpoint": str(ROOT / "weights/avhubert/self_large_vox_433h.pt"),
            "checkpoint_url": "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt",
            "allow_finetuned": True,
            "landmarks": str(ROOT / "weights/avhubert/shape_predictor_68_face_landmarks.dat"),
            "mean_face": str(ROOT / "weights/avhubert/20words_mean_face.npy"),
            "mean_face_url": "https://raw.githubusercontent.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/master/preprocessing/20words_mean_face.npy",
            "device": DEVICE,
            "python": AVH_PYTHON,
            "chunk_frames": 200,
            "overlap_frames": 50,
            "output_stride": 5,
            "max_duration": 60,
            "max_side": 640,
            "workers_per_device": 2,
            "restart_every": 300,  # mở lại worker sau 300 mẫu: tránh RAM tụt dần (part 1, 24/10)
        },
        "xlsr": {
            "kind": "xlsr",
            "model": str(ROOT / "weights/wav2vec2-xls-r-300m"),
            "hf_id": "facebook/wav2vec2-xls-r-300m",
            # Nhóm lớp (hidden_states 1-4, 5-8, 9-12); M1 học trọng số softmax trên các nhóm.
            # Lee et al. 2022 (XLSR-53 đóng băng): lớp 5 tốt nhất cho phát hiện giả.
            "layer_groups": [[1, 4], [5, 8], [9, 12]],
            "device": DEVICE,
        },
        # Chỉ CPU (`cpu_only`): chạy trên phiên Kaggle không GPU, số worker = số nhân CPU.
        # landmarks = stage dò mặt dlib của AV-HuBERT (cùng env / model landmark với "avhubert").
        # Part 1 (24/10): worker dlib rò ~1 GB/giờ mỗi worker; mở lại sau 300 mẫu (~1,5 giờ) giữ
        # RAM tiến trình con dưới ~9 GB. Kết quả không đổi (dò lại cùng frame cho cùng landmark).
        "landmarks": {"kind": "landmarks", "cpu_only": True, "restart_every": 300},
        "mouth": {"kind": "mouth", "crop": 96, "box_max_side": 640, "cpu_only": True},
        "lipinc": {"kind": "lipinc", "box_max_side": 640, "cpu_only": True},
        "wave": {"kind": "wave", "cpu_only": True},
    }
    # Tài sản dựng model: XLS-R gốc cho BL-A fine-tune; checkpoint AVH-Align công bố (train trên
    # 45 k video real AV-1M, commit đã pin) để khởi tạo BL-M và Φ của M2.
    cfg["model_assets"] = {
        "xlsr": cfg["encoders"]["xlsr"]["model"],
        "avh_align_init": str(ROOT / "weights/avh_align/AVH-Align_AV1M.pt"),
        "avh_align_url": "https://raw.githubusercontent.com/bit-ml/AVH-Align/"
        "55401ded7d94a6ad59106c401696d89ce8d16faf/checkpoints/AVH-Align_AV1M.pt",
    }
    return cfg


def bootstrap():
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path[:0] = [str(ROOT / "src")]
