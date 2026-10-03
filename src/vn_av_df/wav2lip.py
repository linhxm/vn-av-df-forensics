"""Adapter following the team's lossless Wav2Lip bridge; official weights only."""

import os
import sys
import tempfile
from pathlib import Path

from vn_av_df.common.runtime import read_json, run, sha, write_json
from vn_av_df.data.media import ffmpeg

REPOSITORY = "https://github.com/Rudrabha/Wav2Lip.git"
GAN_ID = "15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk"


def setup(cfg):
    from vn_av_df.assets import fetch_file

    g = cfg["wav2lip"]
    repo = Path(g["repo"])
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", REPOSITORY, repo])
    revision = run(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    lock = repo / "vn_av_revision.json"
    if lock.exists() and read_json(lock)["commit"] != revision:
        raise ValueError("Generator source changed; use a separate repo directory")
    write_json(lock, {"commit": revision, "repository": REPOSITORY})
    checkpoint = Path(g["checkpoint"])
    if not checkpoint.exists():
        import gdown

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        partial = checkpoint.with_suffix(".partial")
        if not gdown.download(id=GAN_ID, output=str(partial), quiet=False):
            raise RuntimeError(
                "Official Drive download failed; download Wav2Lip-SD-GAN.pt from upstream README to the configured checkpoint"
            )
        os.replace(partial, checkpoint)
    face = repo / "face_detection/detection/sfd/s3fd.pth"
    if not face.exists():
        fetch_file("https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth", face)
    return provenance(cfg)


def provenance(cfg):
    g = cfg["wav2lip"]
    repo = Path(g["repo"])
    required = [
        Path(g["checkpoint"]),
        repo / "inference.py",
        repo / "face_detection/detection/sfd/s3fd.pth",
    ]
    if any(not p.is_file() for p in required):
        raise FileNotFoundError("Wav2Lip assets missing: run generation/00_setup.py")
    return {
        "generator": "wav2lip_gan",
        "version": run(["git", "rev-parse", "HEAD"], cwd=repo).strip(),
        "checkpoint_sha256": sha(required[0]),
        "face_sha256": sha(required[2]),
        "worker_sha256": sha(Path(__file__).with_name("wav2lip_worker.py")),
    }


def synthesize(cfg, video, audio_video, output, frames, width, height):
    g = cfg["wav2lip"]
    with tempfile.TemporaryDirectory(dir=Path(output).parent) as folder:
        temp = Path(folder)
        run(
            [
                ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-i",
                video,
                "-an",
                "-vf",
                f"fps=25,scale={width}:{height}",
                "-frames:v",
                frames,
                "-c:v",
                "ffv1",
                temp / "face.avi",
            ]
        )
        run(
            [
                ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-i",
                audio_video,
                "-vn",
                "-t",
                frames / 25 + 0.2,
                "-af",
                f"atrim=duration={frames / 25},apad=pad_dur=0.2",
                "-ar",
                "16000",
                "-ac",
                "1",
                temp / "audio.wav",
            ]
        )
        # Keep bridge in its own process: upstream imports and global argv never pollute training.
        run(
            [
                g.get("python", sys.executable),
                "-B",
                Path(__file__).with_name("wav2lip_worker.py"),
                "--upstream",
                Path(g["repo"]).resolve(),
                "--checkpoint",
                Path(g["checkpoint"]).resolve(),
                "--device",
                g["device"],
                "--ffmpeg",
                ffmpeg(),
            ],
            cwd=temp,
            timeout=3600,
        )
        run(
            [
                ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-i",
                temp / "generated.mkv",
                "-t",
                frames / 25,
                "-frames:v",
                frames,
                "-c:v",
                "ffv1",
                "-c:a",
                "pcm_s16le",
                "-ar",
                "48000",
                output,
            ]
        )
