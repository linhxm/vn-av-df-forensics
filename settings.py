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
RUN_NAME = "train_all_v005"  # Output: runs/<RUN_NAME>/<model>_seed<seed>/. v004: train_all_v004; v003: train_part1_v003
DEVICE = "cuda"  # Kaggle GPU. Use "cpu" for a local CPU-only environment.

# Generator và hệ giọng: nhóm đầu dùng cho train/validation/test, nhóm sau chỉ ở test.
TRAIN_GENERATORS = ["wav2lip_gan", "musetalk_1_5", "iplap", "videoretalking"]
TEST_ONLY_GENERATORS = ["latentsync_1_5", "diff2lip"]
TRAIN_VOICES = ["vieneu", "f5_vi", "omnivoice", "seedvc"]
TEST_ONLY_VOICES = ["voxcpm2", "vixtts"]
# Đội hình v005 (RESEARCH_WORKLOG/v005_model.md; v004: v004_model.md §7). Baseline đúng code gốc như v003
# (BL-V, BL-A, BL-M, BL-M/sup) + SpeechForensics (không train) + BL-A2 (XLS-R đóng băng); M1 (kiến
# trúc v003) và M2 (giữ nguyên) train lại trên cache / nhãn mới. v005 (chốt 07/11/2026): model chính
# M3-R = trihead_v5 (M3 gọn + học video nén); M3 của v004 (trihead_v4) thành ablation.
# Ghép muộn tính ở evaluate, không train: BL-Ens = max(BL-V, BL-A); BL-Ens-AV = max(BL-M/sup, BL-A2).
MODELS = [
    "lipinc_v2",
    "ssl_aasist",
    "xlsr_frozen",
    "avh_align",
    "avh_align_sup",
    "speechforensics",
    "trihead_v2",
    "avh_align_plus",
    "trihead_v5",
]
# Ablation của M3-R: bỏ cặp sạch ↔ nén (nhất quán), thêm nhánh chất lượng, không học nén; M3 cũ
# (trihead_v4, có nhánh ngữ điệu ↔ đầu). Thử nghiệm 07/11 (part 3, 1.600 step): nhánh ngữ điệu, Φ
# riêng, tham chiếu người không giúp nên ablation v4 cũ (trihead_v4_no_prosody / _no_phi / _abs /
# _ref) chỉ còn trong models.SPECS, chạy tay bằng GROUP lẻ nếu cần.
ABLATIONS = [
    "trihead_v5_no_pair",
    "trihead_v5_quality",
    "trihead_v5_clean",
    "trihead_v4",
]
# Biến thể dữ liệu v005 (RESEARCH_WORKLOG/v005_model.md §8-9), mỗi biến thể một phiên prepare cùng tên
# (landmark dùng lại của bản gốc; trích avhubert / mouth / xlsr trên bản đã xử lý, giữ số frame).
# - c480, c360: nén lại video + tiếng như khi đăng lại mạng xã hội (libx264 + AAC). TRAIN (M3-R theo cặp
#   sạch ↔ bản nén) + test. Thử nghiệm 07-08/11: chỉ học c480 thì real 360p bị báo AI 40 % (nhánh
#   đồng bộ đọc môi kém ở 360p), nên học cả hai mức.
# - pr ("processed real"): video của real / sham / voice qua xử lý lành tính giống đường ống generator
#   (x264 CRF 8 → codec / co giãn → x264 CRF 18), tiếng giữ nguyên, nhãn giữ nguyên: phép thử "model phát
#   hiện AI thật hay dấu vết đường ống" (validation: đường học; test: một nửa 4 họ thường, một nửa 3 họ
#   khác, prepare.processed_real_chain). KHÔNG train: thử 07-08/11, M3-R không học PR đã không bị PR
#   đánh lừa (FPR 0,01-0,02) và học thêm PR không đổi kết quả (2 phép so).
# - w360 (VP9 360p + Opus), c288 (x264 288p): CHỈ test (mức không có khi train: codec khác cùng độ phân
#   giải, độ phân giải thấp hơn); chỉ nén mẫu test.
VARIANTS = {
    "c480": {"name": "c480", "height": 480, "crf": 30, "audio_bitrate": "64k", "preset": "veryfast", "train": True},
    "c360": {"name": "c360", "height": 360, "crf": 35, "audio_bitrate": "32k", "preset": "veryfast", "train": True},
    "pr": {
        "name": "pr",
        "kind": "processed_real",
        "cells": ["real", "sham", "voice"],
        "families": ["mp4v", "mjpeg", "rescale", "x264mid"],
        "heldout_families": ["vp9", "denoise_sharpen", "mpeg2"],
        "splits": ["validation", "test"],
        "train": False,
    },
    "w360": {
        "name": "w360", "height": 360, "video_codec": "libvpx-vp9", "crf": 40,
        "audio_codec": "libopus", "audio_bitrate": "32k", "splits": ["test"], "train": False,
    },
    "c288": {
        "name": "c288", "height": 288, "crf": 37, "audio_bitrate": "24k", "preset": "veryfast",
        "splits": ["test"], "train": False,
    },
}
# Cổng độ phân giải (v005 §9): video có cạnh ngắn dưới mức train thấp nhất (c360) → không xét hình, AI? = audio.
# Đo 08/11 ở 288p (chưa có khi train): mọi model dựa AV-HuBERT AUC ~0,5, báo AI 94-100 % real; có cổng: FPR 0,01-0,03.
VISUAL_MIN_SHORT_SIDE = 360
SEEDS = [42]  # lượt đầu một seed; chốt thiết kế rồi mới chạy [42, 43, 44]
# Bước train v004 (RESEARCH_WORKLOG/v004_model.md §8): **mỗi bước một model** (BL-M đi kèm SF vì SF
# không train), mỗi bước một phiên Kaggle một GPU, để quản lý / bỏ từng model. Thứ tự = ưu tiên:
# "core" là đội hình tối thiểu cho kết luận chính; "optional" bỏ được (mất cổng ghi ở "gates").
# Giờ ○ = phút/epoch part 1 (v003 đo) × 3 part × [số epoch v003 … trần epoch]; bước M3 chạy
# benchmark mọi model trên cache thật trước để chốt lại. Train xong mỗi bước thì test ngay model
# đó trên test cả 3 part; bước "report" (CPU) gộp mọi bước, ghép muộn, cổng, report.md.
TRAIN_GROUPS = {
    "M3-R": {
        "title": "M3-R (v005: M3 gọn + học video nén) + benchmark mọi model",
        "lanes": [["trihead_v5"]],
        "benchmark": True,
        # ○ Cặp sạch + nén: mỗi epoch xử lý mỗi mẫu 2 lần; giây / step như bản không cặp (07/11: 1.075
        # so với 1.042 s / 1.600 step) nên giờ / epoch ≈ 2 lần; dừng ở MAX_HOURS thì phiên sau chạy tiếp.
        "hours": [5, 11],
        "tier": "core",
        "gates": ["mọi cổng R", "V2", "V3", "V6", "V7"],
    },
    "M3-R-nopair": {
        "title": "M3-R bỏ cặp sạch ↔ nén (bản nén lấy ngẫu nhiên, không loss nhất quán)",
        "lanes": [["trihead_v5_no_pair"]],
        "hours": [3, 6.5],
        "tier": "core",
        "gates": ["R6"],
    },
    "M2": {
        "title": "M2 (AVH-Align+, giữ nguyên v003)",
        "lanes": [["avh_align_plus"]],
        "hours": [0.7, 1.9],
        "tier": "core",
        "gates": ["V3"],
    },
    "BL-M": {
        "title": "BL-M (AVH-Align không giám sát) + SpeechForensics (không train)",
        "lanes": [["avh_align", "speechforensics"]],
        "hours": [0.3, 0.5],
        "tier": "core",
        "gates": ["V7"],
    },
    "BL-Msup": {
        "title": "BL-M/sup (AVH-Align có giám sát)",
        "lanes": [["avh_align_sup"]],
        "hours": [3.3, 10],
        "tier": "core",
        "gates": ["V2 (BL-Ens-AV)"],
    },
    "BL-A2": {
        "title": "BL-A2 (XLS-R đóng băng)",
        "lanes": [["xlsr_frozen"]],
        "hours": [1, 2],
        "tier": "core",
        "gates": ["V2 (BL-Ens-AV)", "V6"],
    },
    "BL-V": {
        "title": "BL-V (LIPINC-V2)",
        "lanes": [["lipinc_v2"]],
        "hours": [1.4, 5],
        "tier": "core",
        "gates": ["BL-Ens"],
    },
    "M1": {
        "title": "M1 (kiến trúc v003, có tham chiếu người): 2-3 phiên; hoãn, xem kết quả M3 rồi quyết",
        "lanes": [["trihead_v2"]],
        "hours": [12.3, 24.5],
        "tier": "deferred",
        "gates": ["V1"],
    },
    "M3-R-quality": {
        "title": "M3-R thêm nhánh chất lượng (FiLM theo độ phân giải miệng + đầu phụ 'bị nén')",
        "lanes": [["trihead_v5_quality"]],
        "hours": [5, 11],
        "tier": "optional",
        "gates": ["R5"],
    },
    "M3-R-clean": {
        "title": "M3-R không học nén (chỉ bản gốc)",
        "lanes": [["trihead_v5_clean"]],
        "hours": [3, 6.5],
        "tier": "optional",
        "gates": ["R4"],
    },
    "M3": {
        "title": "M3 của v004 (trihead_v4, có nhánh ngữ điệu ↔ đầu)",
        "lanes": [["trihead_v4"]],
        "hours": [3, 7],
        "tier": "optional",
        "gates": ["R7"],
    },
    "BL-A": {
        "title": "BL-A (SSL-AASIST fine-tune XLS-R; v003 không học được): 1-4 phiên",
        "lanes": [["ssl_aasist"]],
        "hours": [10, 43],
        "tier": "optional",
        "gates": ["BL-Ens"],
    },
}
# Phiên Kaggle T4 ×2 (1 giờ T4 ×2 trừ 1 giờ quota: thông báo T4 của Kaggle): hai hàng song song,
# mỗi hàng một GPU, các bước trong một hàng chạy nối tiếp; model nào xong thì test ngay. Bỏ bước
# nào thì xoá khỏi hàng (notebook: DROP). P5: PREVIOUS_RUN = Output P4 để BL-A (và M3 nếu dở) chạy tiếp
# từ last.pt. M1 không nằm trong phiên nào: user chốt 03/11 hoãn, có kết quả M3 rồi mới quyết,
# chạy bằng GROUP = "M1". Giờ ○ ở RESEARCH_WORKLOG/v004_model.md §8.2.
TRAIN_SESSIONS = {
    "P1": [["M3-R"], ["M3-R-nopair"]],
    "P2": [["BL-Msup"], ["BL-A2", "BL-V", "M2", "BL-M"]],
    "P3": [["M3-R-quality"], ["M3-R-clean"]],
    "P4": [["BL-A"], ["M3"]],
    "P5": [["BL-A"], ["M3"]],
}
# Đội hình đầy đủ: notebook in model nào chưa train xong; report gồm các model đã có (cổng thiếu
# model thì bỏ qua).
TEST_MODELS = MODELS + ABLATIONS
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
            # Chọn checkpoint và dừng sớm (mọi model, cùng một luật): "auc" = trung bình AUC validation
            # của các tác vụ mà model có đầu ra (model 3 đầu: visual / audio / mismatch; BL-V: visual;
            # BL-A, BL-A2: audio; BL-M/sup: AI?; M2: AI? + mismatch); model chỉ học từ real (BL-M) và
            # validation thiếu một lớp thì theo val loss. v003 chọn theo val loss: loss nhiễu làm
            # BL-M/sup, BL-V dừng ở epoch AUC thấp hơn 0,01-0,04 (v004_model.md §8).
            "monitor": "auc",
            # Model của mình (M1, M2, M3, BL-A2; baseline giữ scheduler của code gốc): giảm lr ×0,1
            # sau 2 epoch không tốt hơn, dừng sau 5. v003: lr cố định, val loss M1 nhảy 0,90 → 1,23.
            "scheduler": "plateau",
            "plateau_patience": 2,
            # Dịch tiếng so với hình k frame khi train (dương = tiếng trễ). v003: [−1, +2], nhãn giữ
            # nguyên (docs/v003_data.md §4.3.4). v004: [−2, +5] phủ trễ tự nhiên của real YouTube
            # (p10-p90 +1…+4 frame, v003_data §5.5) và trễ của generator (+1…+3, A11 / A12); nhãn
            # mismatch tính lại sau khi trượt (training/sync_labels.py) nên real trễ nhiều là
            # "lệch, không AI" → đầu visual không học được lối tắt "trễ ⇒ giả".
            "audio_lag_aug": [-2, 5],
            # Mỗi người 15-20 clip: lấy mẫu train theo 1 / số clip của người đó để mỗi người đóng
            # góp như nhau (docs/v003_model.md §7).
            "speaker_balance": True,
            # Học video nén (cfg["variants"], v005): model không học nhất quán lấy bản nén
            # với xác suất này (mọi đặc trưng của mẫu cùng lấy từ bản nén); M3-R học theo cặp.
            "variant_p": 0.5,
        },
        "bootstrap_replicates": 1000,
        # Nhãn mismatch (training/sync_labels.py, RESEARCH_WORKLOG/v004_model.md §5):
        # "measured" = lệch nội dung theo cách tạo + ô lip-sync đo tại lag 0 so với real (cần
        # <part>/sync_syncnet.json từ training/01a_syncnet.py); "mask_lipsync" = dự phòng không cần
        # đo (ô lip-sync → mask); "construction" = như v003.
        # User chốt 03/11: một thước SyncNet, một ngưỡng p95 của real train. Cổng độ nhọn hạ về mức
        # thấp nhất của real train nên train không còn mask; validation, test còn 14 mẫu lip-sync
        # mask vì 3 clip real gốc nhọn dưới mức đó.
        "mismatch_labels": {
            "mode": "measured",
            "meters": ["syncnet"],
            "near_shift": 1,
            "negative_quantile": 0.95,
            "positive_quantile": 0.95,
            "sharpness_quantile": 0.0,
        },
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
            # Nhóm lớp hidden_states; model học trọng số softmax trên các nhóm. v003: 1-4, 5-8, 9-12
            # (Lee et al. 2022: lớp 5 tốt nhất với XLSR-53). v004 phủ cả 24 lớp: lớp giữa / trộn
            # nhiều lớp tổng quát tốt ngoài miền (Serrano et al. 2025; SLS, ACM MM 2024); giọng chưa
            # thấy là điểm yếu nhất của M1 (0,889). Cache ~5/3 lần bản 3 nhóm.
            "layer_groups": [[1, 4], [5, 8], [9, 12], [13, 16], [17, 24]],
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
    # Đặc tả biến thể (train hay chỉ test, PR): train.py / evaluate.py đọc; thư mục cache ở cfg["variants"].
    cfg["variant_specs"] = VARIANTS
    cfg["visual_min_short_side"] = VISUAL_MIN_SHORT_SIDE
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
