"""Worker AV-HuBERT tách môi trường fairseq khỏi FATE/transformers."""

import sys
import tempfile
from pathlib import Path

from vn_av_df.common.runtime import fingerprint, read_json, require_file, run, sha, write_json


class AVHubertVideoEncoder:
    """Cache native 25 Hz có PTS/mask; strict signature cả upstream lẫn preprocessing."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        repo = Path(cfg["repo"])
        paths = {
            name: require_file(cfg[name], name) for name in ("checkpoint", "landmarks", "mean_face")
        }
        for name in (
            "hubert.py",
            "hubert_pretraining.py",
            "utils.py",
            "preparation/align_mouth.py",
        ):
            paths[name] = require_file(repo / "avhubert" / name, name)
        paths["worker"] = Path(__file__).with_name("avhubert_worker.py")
        paths["adapter"] = Path(__file__)
        paths["media"] = Path(__file__).parents[1] / "data/media.py"
        self.signature = fingerprint(
            dict(
                assets={k: sha(v) for k, v in paths.items()},
                options={
                    k: cfg.get(k, default)
                    for k, default in (
                        ("chunk_frames", 200),
                        ("overlap_frames", 50),
                        ("max_duration", 60),
                        ("max_side", 640),
                        ("output_stride", 5),
                    )
                },
                format="avhubert-native-v1",
            )
        )

    def extract(self, row, output):
        """Nhận media đã render; worker không tải model hoặc tự chỉnh lệch tiếng."""
        output = Path(output).resolve()
        video = require_file(row["video"], "video")
        source = fingerprint({"assets": {"video": sha(video)}, "variant": {"kind": "clean"}})
        if output.is_file() and output.with_suffix(".json").is_file():
            meta = read_json(output.with_suffix(".json"))
            if (
                meta.get("feature_signature") == self.signature
                and meta.get("source_fingerprint") == source
                and meta.get("feature_sha256") == sha(output)
            ):
                return meta
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as folder:
            request = Path(folder) / "request.json"
            write_json(
                request,
                dict(
                    config=self.cfg,
                    video=str(video),
                    output=str(output),
                    signature=self.signature,
                    source_fingerprint=source,
                ),
            )
            run(
                [
                    self.cfg.get("python", sys.executable),
                    "-B",
                    Path(__file__).with_name("avhubert_worker.py"),
                    request,
                ],
                timeout=3600,
            )
        return read_json(output.with_suffix(".json"))
