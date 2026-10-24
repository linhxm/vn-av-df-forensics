"""Worker AV-HuBERT tách môi trường fairseq khỏi transformers của môi trường chính.

Hai stage (xem avhubert_worker.py): "landmarks" (dlib, CPU, không cần checkpoint) và "full"
(model; đọc landmark có sẵn nếu prepare đưa `landmarks`, không thì tự dò như trước — demo).
"""

import json
import sys
import threading
import time
from pathlib import Path

from vn_av_df import worker
from vn_av_df.common.runtime import fingerprint, read_json, require_file, sha

FORMATS = {"full": "avhubert-native-v2", "landmarks": "avhubert-landmarks-v1"}


class AVHubertVideoEncoder:
    """Cache native 25 Hz có PTS/mask; strict signature cả upstream lẫn preprocessing."""

    # Mỗi worker nạp checkpoint 5,7 GB: mở lần lượt (kể cả khi mở lại) để không vượt RAM Kaggle.
    starting = threading.Lock()

    def __init__(self, cfg, stage="full"):
        if stage not in FORMATS:
            raise ValueError(f"Unknown AV-HuBERT stage {stage}")
        self.stage = stage
        self.cfg = {**cfg, "stage": stage}
        self.worker = None
        # Part 1 trên Kaggle: RAM trống tụt 26 → 0,4 GB sau ~7.500 mẫu, worker bị giết (-9) / treo.
        # Mở lại worker sau mỗi `restart_every` mẫu để trả RAM (đặc trưng không đổi: model eval).
        self.calls, self.restart_every = 0, int(cfg.get("restart_every", 300))
        # Stage landmarks chỉ phụ thuộc dlib + mean face: không cần (và không tải) checkpoint.
        names = (
            ("landmarks", "mean_face")
            if stage == "landmarks"
            else ("checkpoint", "landmarks", "mean_face")
        )
        paths = {name: require_file(cfg[name], name) for name in names}
        if stage == "full":
            repo = Path(cfg["repo"])
            for name in (
                "hubert.py",
                "hubert_pretraining.py",
                "hubert_asr.py",
                "decoder.py",
                "utils.py",
                "preparation/align_mouth.py",
            ):
                paths[name] = require_file(repo / "avhubert" / name, name)
        paths["worker"] = Path(__file__).with_name("avhubert_worker.py")
        paths["adapter"] = Path(__file__)
        paths["media"] = Path(__file__).parents[1] / "data/media.py"
        options = (
            ("max_duration", 60),
            ("max_side", 640),
            *(
                (("chunk_frames", 200), ("overlap_frames", 50), ("output_stride", 5))
                if stage == "full"
                else ()
            ),
        )
        self.signature = fingerprint(
            dict(
                assets={k: sha(v) for k, v in paths.items()},
                options={k: cfg.get(k, default) for k, default in options},
                format=FORMATS[stage],
            )
        )
        self.name = "AV-HuBERT worker" if stage == "full" else "Landmark worker (dlib)"

    def start(self):
        """Mở worker nếu chưa chạy. prepare gọi lần lượt từng encoder trước khi trích: 4 worker
        stage full cùng nạp checkpoint 5,7 GB một lúc vượt RAM Kaggle (~30 GB)."""
        with self.starting:
            self._start()

    def _start(self):
        if self.worker is None or not self.worker.alive:
            again = self.worker is not None
            device = self.cfg.get("device", "cpu") if self.stage == "full" else "cpu"
            what = "checkpoint + dlib" if self.stage == "full" else "dlib"
            print(
                f"{self.name} ({device}): {'mở lại sau khi worker chết' if again else 'khởi động'}"
                f", nạp {what}…",
                flush=True,
            )
            began = time.monotonic()
            self.worker = worker.Worker(
                [
                    self.cfg.get("python", sys.executable),
                    "-B",
                    Path(__file__).with_name("avhubert_worker.py"),
                    "--serve",
                    json.dumps(self.cfg, sort_keys=True),
                ]
            )
            print(
                f"{self.name} ({device}): sẵn sàng sau {time.monotonic() - began:.0f} s", flush=True
            )

    def extract(self, row, output):
        """Nhận media đã render; worker không tải model hoặc tự chỉnh lệch tiếng.

        row["landmarks"] (stage full): file của stage landmarks; chữ ký cache gồm cả file đó."""
        output = Path(output).resolve()
        video = require_file(row["video"], "video")
        source = fingerprint({"assets": {"video": sha(video)}, "variant": {"kind": "clean"}})
        signature = self.signature
        landmarks = row.get("landmarks") if self.stage == "full" else None
        if landmarks and Path(landmarks).is_file():
            meta = read_json(Path(landmarks).with_suffix(".json"))
            signature = fingerprint([self.signature, meta["feature_sha256"]])
        else:
            landmarks = None
        if output.is_file() and output.with_suffix(".json").is_file():
            meta = read_json(output.with_suffix(".json"))
            if (
                meta.get("feature_signature") == signature
                and meta.get("source_fingerprint") == source
                and meta.get("feature_sha256") == sha(output)
            ):
                return meta
        output.parent.mkdir(parents=True, exist_ok=True)
        # Worker sống suốt lượt trích (nạp một lần), mỗi encoder một worker; worker chết thì lần
        # sau tự mở lại; quá restart_every mẫu thì mở lại để trả RAM.
        if self.worker is not None and self.calls >= self.restart_every:
            print(f"{self.name}: mở lại định kỳ sau {self.calls} mẫu để trả RAM", flush=True)
            self.close()
        self.start()
        self.calls += 1
        self.worker.call(
            dict(
                video=str(video),
                output=str(output),
                signature=signature,
                source_fingerprint=source,
                landmarks=str(landmarks) if landmarks else None,
            ),
            timeout=3600,
        )
        return read_json(output.with_suffix(".json"))

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None
            self.calls = 0
