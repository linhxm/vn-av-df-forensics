"""Explicit, resumable downloads. Record exact revisions and hashes for reproducibility."""

import os
import shutil
import urllib.request
from pathlib import Path

from vn_av_df.common.runtime import read_json, run, sha, write_json

MEDIA = {
    "face": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}
FATE_REVISION = "beae95aeb6f72cf1751d06d1428931016a7a1867"
HF_REVISIONS = {
    "facebook/pe-av-small": "dd050762bb9704ae9cd996ca45532a98f81d817e",
    "Guan123/fate": "8463ab93a644a22bd85db91e7e77d99ebe1ec5e0",
    "facebook/dinov2-small": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
}


def hf_snapshot(name, target, patterns):
    """Tải đúng revision đã pin; ghi revision trước để phiên bị ngắt resume cùng snapshot."""
    from huggingface_hub import snapshot_download

    target = Path(target)
    revision_file = target / "vn_av_revision.json"
    revision = (
        read_json(revision_file)["revision"] if revision_file.exists() else HF_REVISIONS[name]
    )
    write_json(revision_file, {"repo_id": name, "revision": revision})
    print(f"Downloading {name}@{revision}", flush=True)
    try:
        snapshot_download(
            name, revision=revision, local_dir=target, max_workers=2, allow_patterns=patterns
        )
    except Exception as exc:
        raise RuntimeError(
            f"Download {name} failed. Rerun training/00_setup.py "
            f"on a working Internet connection; partial files retained. {exc}"
        ) from exc
    return {"repo_id": name, "revision": revision}


def pinned_repository(url, target, recursive=False):
    """Lần đầu ghi commit, lần sau yêu cầu đúng commit; không pull ngầm."""
    target = Path(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", *(["--recurse-submodules"] if recursive else []), url, target])
    commit = run(["git", "rev-parse", "HEAD"], cwd=target).strip()
    lock = target / "vn_av_revision.json"
    if lock.exists() and read_json(lock)["commit"] != commit:
        raise ValueError("Upstream commit changed; use new source/cache directory")
    write_json(lock, dict(repository=url, commit=commit))
    return commit


def pinned_snapshot(repo_id, target, patterns):
    """Resolve revision một lần trước download để resume cùng tài sản."""
    from huggingface_hub import HfApi, snapshot_download

    target = Path(target)
    lock = target / "vn_av_revision.json"
    revision = read_json(lock)["revision"] if lock.exists() else HfApi().model_info(repo_id).sha
    write_json(lock, dict(repo_id=repo_id, revision=revision))
    snapshot_download(
        repo_id, revision=revision, local_dir=target, allow_patterns=patterns, max_workers=2
    )


def setup_avhubert(encoder):
    """Chuẩn bị AV-HuBERT Base và landmark; không cài fairseq vào env FATE."""
    import bz2

    pinned_repository(
        "https://github.com/facebookresearch/av_hubert.git", encoder["repo"], recursive=True
    )
    result = {
        "checkpoint": fetch_file(
            "https://dl.fbaipublicfiles.com/avhubert/model/lrs3/clean-pretrain/base_lrs3_iter4.pt",
            encoder["checkpoint"],
        )
    }
    landmark = Path(encoder["landmarks"])
    compressed = landmark.with_suffix(".dat.bz2")
    fetch_file("https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", compressed)
    if not landmark.exists():
        from vn_av_df.common.runtime import atomic_bytes

        atomic_bytes(landmark, bz2.decompress(compressed.read_bytes()))
    result["mean_face"] = fetch_file(encoder["mean_face_url"], encoder["mean_face"])
    return result


def fetch_file(url, destination):
    destination = Path(destination)
    record = destination.with_suffix(destination.suffix + ".download.json")
    if destination.exists():
        if record.exists() and read_json(record)["sha256"] == sha(destination):
            return read_json(record)
        raise ValueError(f"Unverified existing asset: {destination}; use a new asset path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        if partial.stat().st_size < 1024:
            raise ValueError(f"Invalid model download: {url}")
        os.replace(partial, destination)
        result = {"url": url, "sha256": sha(destination), "bytes": destination.stat().st_size}
        write_json(record, result)
        return result
    finally:
        partial.unlink(missing_ok=True)


def setup_assets(cfg, only="all"):
    """Setup tất cả encoder được chọn; không tải các nhánh ablation không dùng."""
    if cfg.get("methods"):
        from vn_av_df.features.registry import method_config

        results, only = {}, cfg.get("prepare_encoders")  # None = mọi encoder kiến trúc cần.
        for name in cfg["architectures"]:
            spec = cfg["methods"][name]
            scoped = method_config(cfg, name)
            scoped.pop("methods", None)
            for key in filter(None, (spec["encoder"], spec.get("artifact_encoder"))):
                if key not in results and (only is None or key in only):
                    results[key] = setup_assets({**scoped, "encoder": cfg["encoders"][key]}, only)
        return results
    if cfg["encoder"].get("kind") == "avhubert":
        return setup_avhubert(cfg["encoder"])
    if cfg["encoder"].get("kind") == "dinov2":
        return hf_snapshot(
            "facebook/dinov2-small",
            cfg["encoder"]["model"],
            ["config.json", "model.safetensors", "preprocessor_config.json"],
        )
    result = {}
    if only in ("all", "media"):
        target = cfg["encoder"]["face_model"]
        result["face"] = fetch_file(MEDIA["face"], target)
    if only in ("all", "fate", "source"):
        encoder = cfg["encoder"]
        repo = Path(encoder["repo"])
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "--depth", "1", "https://github.com/guankaisi/FATE.git", repo])
            run(["git", "fetch", "--depth", "1", "origin", FATE_REVISION], cwd=repo)
            run(["git", "checkout", "--detach", FATE_REVISION], cwd=repo)
        if not (repo / "models/pe_av/modeling_pe_audio_video.py").is_file():
            raise ValueError(f"Incomplete FATE source: {repo}")
        commit = run(["git", "rev-parse", "HEAD"], cwd=repo).strip()
        if commit != FATE_REVISION:
            raise ValueError(
                "FATE source differs from the tested revision; use a new encoder.repo path"
            )
        lock = repo / "vn_av_revision.json"
        if lock.exists() and read_json(lock)["commit"] != commit:
            raise ValueError("FATE revision changed; use a separate source/cache directory")
        write_json(lock, {"commit": commit, "repository": "https://github.com/guankaisi/FATE"})
        result["fate_source"] = commit
        if only != "source":
            for key, name in (("base", "facebook/pe-av-small"), ("adapter", "Guan123/fate")):
                result[key] = hf_snapshot(
                    name, encoder[key], ["*.json", "*.safetensors", "*.txt", "*.model"]
                )
    return result
