"""Cấu hình mặc định cho các file .py ở local.

Data/review: chọn PART trong data_settings.py. Generate/train trên Kaggle: chỉnh
các cell cấu hình trong notebook, không cần sửa file này rồi upload lại.
config() lấy giá trị hiện tại; trong notebook chỉ gọi nó sau khi chọn đường dẫn/model.
RUN_NAME là folder kết quả thí nghiệm, không phải tên dataset hay tên kiến trúc.
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
# DATASET_PARTS = ["vn-av-df-data-part1", "vn-av-df-data-part2"]
# DATASET_PARTS = "all"  # Chỉ các part đã tải/attach, không tự download.
# Split train/validation/test đã gán ở 05_export (xem data_settings.py).
RUN_NAME = "train_part1_run01"  # Output: runs/train_part1_run01/<model>_seed<seed>/.
DEVICE = "cuda"  # Kaggle GPU. Use "cpu" for a local CPU-only environment.
RESUME = False
# B-FATE, B-AVH (cùng AV-HuBERT, không mô hình hóa consistency) và P2 đề xuất.
ARCHITECTURES = ["fate_gru", "avh_tcn", "p2_syncartifact"]
# ARCHITECTURES = ["fate_gru"]  # Chỉ một kiến trúc.
# Ablation P2 (dùng chung cache AV-HuBERT/DINOv2):
# ARCHITECTURES = ["avh_realrecon", "p2_sync_only", "p2_artifact_only", "p2_concat"]
SEEDS = [42, 43, 44]  # Use [42] and EPOCHS=2 for a software smoke run.
EPOCHS = 20
HELD_OUT_GENERATORS = ["musetalk_1_5"]
AVH_PYTHON = os.environ.get(
    "VN_AVH_PYTHON",
    str(ROOT / ".venv-avhubert" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
)
MUSETALK_PYTHON = os.environ.get(
    "VN_MUSETALK_PYTHON",
    str(ROOT / ".venv-musetalk" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
)
CHECKPOINT = None  # None dùng DEMO_METHOD + seed đầu trong training.json, không xếp hạng model.
DEMO_METHOD = ARCHITECTURES[0]  # Hoặc chỉ rõ "avh_realrecon" nếu đã train model đó.


def config():
    """Cấu hình dùng chung cho Run File và notebook; không hardcode thuật toán trong notebook."""
    cfg = {
        "seed": 42,
        "clean_dataset": str(CLEAN_DATASET),
        "generated_dataset": str(ROOT / "datasets" / DATASET_NAME / GENERATION_NAME),
        "data_part": PART,
        "dataset": str(DATASET),
        "dataset_parts": DATASET_PARTS,
        "plan": str(ROOT / "generation/plans" / f"{GENERATION_NAME}.json"),
        "generation": {
            "clips_per_split": 2,
            "partial_seconds": [0.4, 0.8, 1.6, 2.4],
            "crf": 18,
            "max_side": 640,
            # Thiết kế 2×2: donor = tiếng thật khác của cùng người (mối đe dọa chính, có lệch);
            # source = vẽ theo đúng tiếng gốc (chỉ artifact), dùng làm đối chứng ở val/test.
            "fake_audio_modes_by_split": {
                "train": ["donor"],
                "validation": ["donor", "source"],
                "test": ["donor", "source"],
            },
            "include_sham": True,  # Sham = lệch A/V không AI, nhãn AI 0; cần cùng speaker_id.
            "generators_by_split": {
                "train": ["wav2lip_gan"],
                "validation": ["wav2lip_gan"],
                "test": ["wav2lip_gan", "musetalk_1_5"],
            },
        },
        "wav2lip": {
            "repo": str(ROOT / "external/Wav2Lip"),
            "checkpoint": str(ROOT / "weights/wav2lip/Wav2Lip-SD-GAN.pt"),
            "device": DEVICE,
            "python": sys.executable,
        },
        "encoder": {
            "kind": "fate",
            "repo": str(ROOT / "external/FATE"),
            "base": str(ROOT / "weights/pe-av-small"),
            "adapter": str(ROOT / "weights/fate"),
            "face_model": str(ROOT / "weights/media/yunet.onnx"),
            "device": DEVICE,
            "precision": "float16" if DEVICE == "cuda" else "float32",
            "window_s": 2.0,
            "step_s": 0.2,
            "temporal_bins": 8,
            "min_coverage": 0.75,
            "max_duration": 60,
            "max_side": 640,
        },
        "device": DEVICE,
        "architectures": ARCHITECTURES,
        "seeds": SEEDS,
        "held_out_generators": HELD_OUT_GENERATORS,
        "cache": str(ROOT / "cache" / DATASET_NAME),
        "runs": str(ROOT / "runs" / RUN_NAME),
        "checkpoint": str(CHECKPOINT) if CHECKPOINT else None,
        "training": {
            "epochs": EPOCHS,
            "lr": 0.0003,
            "weight_decay": 0.0001,
            "hidden": 128,
            "dropout": 0.1,
            "patience": 5,
            "top_fraction": 0.1,
            "torch_threads": 2,
            "balance_parents": True,
            "artifact_aux_weight": 0.5,  # P2: head phụ artifact học cùng nhãn AI theo ô.
        },
        "demo_output": str(ROOT / "outputs/demo"),
    }
    cfg["encoders"] = {
        "fate": cfg["encoder"],
        "avhubert": {
            "kind": "avhubert",
            "repo": str(ROOT / "external/av_hubert"),
            "checkpoint": str(ROOT / "weights/avhubert/base_lrs3_iter4.pt"),
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
        },
        "dinov2": {
            "kind": "dinov2",
            "model": str(ROOT / "weights/dinov2-small"),  # facebook/dinov2-small, ViT-S/14.
            "device": DEVICE,
            "precision": "float16" if DEVICE == "cuda" else "float32",
            "crop_size": 224,  # Crop miệng RGB, bội số 14.
            "batch_size": 64,
            "max_duration": 60,
            "max_side": 640,  # Phải trùng AV-HuBERT để hộp miệng khớp frame.
        },
    }
    cfg["methods"] = {
        "fate_gru": {"encoder": "fate", "architecture": "gru"},
        "avh_tcn": {"encoder": "avhubert", "architecture": "tcn"},
        "avh_realrecon": {"encoder": "avhubert", "architecture": "realrecon"},
    }
    p2 = {"encoder": "avhubert", "artifact_encoder": "dinov2"}
    cfg["methods"].update(
        p2_syncartifact={**p2, "architecture": "syncartifact"},  # P2 đầy đủ, gate.
        p2_concat={**p2, "architecture": "syncartifact_concat"},
        p2_sync_only={**p2, "architecture": "sync_only"},
        p2_artifact_only={**p2, "architecture": "artifact_only"},
        p2_sync_seen_fake={**p2, "architecture": "syncartifact_unfrozen"},  # Sync học cả fake.
    )
    cfg["reconstruction"] = {"epochs": 20, "lr": 3e-4, "weight_decay": 1e-4, "patience": 5}
    # Stage S của P2: real (0), sham và real dịch lệch tiếng ±3-15 frame (1); không thấy fake.
    cfg["sync"] = {
        "epochs": 20,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "patience": 5,
        "shift_frames": [3, 15],
        "shift_probability": 0.5,
    }
    cfg["musetalk"] = {
        "repo": str(ROOT / "external/MuseTalk"),
        "python": MUSETALK_PYTHON,
        "device": DEVICE,
        "batch_size": 4,
        "seed": 42,
    }
    cfg["demo_method"] = DEMO_METHOD
    cfg["bootstrap_replicates"] = 1000
    return cfg


def bootstrap():
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data_pipeline/src")]
