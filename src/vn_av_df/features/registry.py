"""Chọn encoder/cache theo phương pháp; mọi model dùng cùng manifest đã khóa."""

import copy
from pathlib import Path

import numpy as np

SYNC_ARTIFACT_MODELS = {
    "syncartifact",
    "syncartifact_concat",
    "syncartifact_unfrozen",
    "sync_only",
    "artifact_only",
}
NATIVE_MODELS = {"realrecon", *SYNC_ARTIFACT_MODELS}


def method_config(cfg, name):
    """Legacy configs còn chạy; config mới tách method name khỏi head type."""
    result = copy.deepcopy(cfg)
    spec = cfg.get("methods", {}).get(name)
    if spec:
        result["encoder"] = copy.deepcopy(cfg["encoders"][spec["encoder"]])
        result["cache"] = str(Path(cfg["cache"]) / spec["encoder"])
        result["model_architecture"] = spec["architecture"]
        if spec.get("artifact_encoder"):
            # P2: cache artifact riêng, đọc hộp miệng từ cache encoder chính cùng sample.
            result["artifact_encoder"] = {
                **copy.deepcopy(cfg["encoders"][spec["artifact_encoder"]]),
                "box_cache": result["cache"],
            }
            result["artifact_cache"] = str(Path(cfg["cache"]) / spec["artifact_encoder"])
    else:
        result["model_architecture"] = name
    result["method_name"] = name
    return result


def combined_signature(primary, artifact=None):
    """Checkpoint P2 phụ thuộc cả hai encoder; model khác giữ nguyên signature cũ."""
    if artifact is None:
        return primary
    from vn_av_df.common.runtime import fingerprint

    return fingerprint({"primary": primary, "artifact": artifact})


def artifact_arrays(path, meta, frames):
    """Đặc trưng DINOv2 native 25 Hz phải khớp đúng số frame của cache AV-HuBERT."""
    with np.load(path, allow_pickle=False) as z:
        features = z["artifact"].astype(np.float32)
        valid = z["artifact_valid"].copy()
    if features.ndim != 2 or len(features) != frames or valid.shape != (frames,):
        raise ValueError("Artifact cache differs from the AV-HuBERT timeline")
    if meta["step_s"] != 0.04:
        raise ValueError("Artifact cache must be native 25 Hz")
    return features, valid


def make_encoder(cfg):
    """Khởi tạo đúng backbone, tải weight chỉ ở bước setup tường minh."""
    if cfg.get("kind", "fate") == "avhubert":
        from vn_av_df.features.avhubert import AVHubertVideoEncoder

        return AVHubertVideoEncoder(cfg)
    if cfg.get("kind") == "dinov2":
        from vn_av_df.features.dinov2 import DinoMouthEncoder

        return DinoMouthEncoder(cfg)
    if cfg.get("kind", "fate") != "fate":
        raise ValueError("Unknown encoder kind")
    from vn_av_df.features.fate import FATEVideoEncoder

    return FATEVideoEncoder(cfg)


def feature_arrays(path, meta, architecture):
    """Native AVH: P1 nhận 25 Hz; baseline nhận mean 5 frame cùng ô 0,2s."""
    with np.load(path, allow_pickle=False) as z:
        a, v = z["audio"].copy(), z["visual"].copy()
        mask = (z["audio_valid"] & z["visual_valid"]).copy()
        times = z["times_s"].copy()
    if (
        a.ndim != 2
        or v.ndim != 2
        or len(a) != len(v)
        or len(a) != len(times)
        or mask.shape != times.shape
    ):
        raise ValueError("Cache dimensions differ")
    if not len(times) or times[0] != 0 or not np.allclose(np.diff(times), meta["step_s"]):
        raise ValueError("Cache timeline differs")
    stride = int(meta.get("output_stride", 1))
    if stride < 1:
        raise ValueError("Invalid output stride")
    output_meta = {**meta, "step_s": meta["step_s"] * stride, "native_stride": stride}
    if stride > 1 and architecture not in NATIVE_MODELS:
        a = np.stack([a[i : i + stride].mean(0) for i in range(0, len(a), stride)])
        v = np.stack([v[i : i + stride].mean(0) for i in range(0, len(v), stride)])
        mask = np.array([mask[i : i + stride].all() for i in range(0, len(mask), stride)])
        output_meta["native_stride"] = 1
    return (a, v, mask), times[::stride], output_meta
