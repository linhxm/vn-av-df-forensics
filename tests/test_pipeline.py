"""Benchmark v003 end-to-end trên dữ liệu tổng hợp (ASR / TTS / generator giả lập, CPU)."""

import copy
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from vn_av_df.common.runtime import run, sha, write_json
from vn_av_df.data.media import ffmpeg
from vn_av_df.dataset import validate_rows
from vn_av_df.generation.cells import labels
from vn_av_df.training import models as zoo

SPLITS = {"train": ["s1", "s2"], "validation": ["s3", "s4"], "test": ["s5", "s6", "s7", "s8", "s9"]}
# XLS-R tí hon cho BL-A (fine-tune) trong test: cùng kiến trúc Wav2Vec2, vài nghìn tham số.
TINY_XLSR = dict(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    conv_dim=(16, 16, 16, 16, 16, 16, 16),
    conv_stride=(5, 2, 2, 2, 2, 2, 2),
    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
    num_conv_pos_embeddings=16,
    num_conv_pos_embedding_groups=2,
)


def make_clip(path, seconds=6, freq=440, size="320x240"):
    path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg(),
            *("-nostdin", "-v", "error", "-y", "-f", "lavfi"),
            *("-i", f"testsrc2=s={size}:r=25:d={seconds},hue=h={freq % 360}"),
            *("-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=48000:duration={seconds}"),
            *(
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                path,
            ),
        ]
    )
    return path


def make_clean(root, clips_per_speaker=2):
    from vn_av_data.contract import SCHEMA

    rows, freq = [], 300
    for split, speakers in SPLITS.items():
        for k, speaker in enumerate(speakers):
            for i in range(clips_per_speaker):
                freq += 37
                cid = f"clip_{speaker}_{i}"
                video = make_clip(root / "clips" / f"{cid}.mp4", 6, freq)
                rows.append(
                    {
                        "clip_id": cid,
                        "video": f"clips/{cid}.mp4",
                        "sha256": sha(video),
                        "source_id": f"yt_{speaker}_v{i % 2}",
                        "speaker_id": speaker,
                        "gender": "M" if k % 2 else "F",
                        "split": split,
                        "duration_s": 6.0,
                        "source_start_s": 10.0 * i,
                        "source_end_s": 10.0 * i + 6.0,
                        "review_decision": "keep",
                        "sync_status": "reviewed_match",
                        "relation_annotations": {},
                    }
                )
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    write_json(
        root / "dataset_info.json",
        {
            "schema_version": SCHEMA,
            "dataset_id": "vn-test-part1",
            "clips": len(rows),
            "manifest_sha256": sha(root / "manifest.jsonl"),
        },
    )
    return rows


def options():
    gens_train, gens_test = ["g_a", "g_b", "g_c"], ["g_a", "g_b", "g_c", "g_new"]
    return {
        "generators": {"train": gens_train, "validation": gens_train, "test": gens_test},
        "voices": {
            "train": ["vieneu", "f5_vi"],
            "validation": ["vieneu", "f5_vi"],
            "test": ["vieneu", "f5_vi", "voxcpm2"],
        },
        "inpaint": {"test": "f5_inpaint"},
        "generators_per_clip": {"test": 2},  # test: 2 trong 4 generator mỗi clip
        "partial_range_s": [0.5, 2.0],
        "edge_margin_s": 0.5,
        "donor_fit": {"shorter_s": 1.0, "min_pause_s": 0.02},
        "blend_probability": 0.5,
        "blend_frames": [2, 6],
        "max_side": 320,
        "crf": 23,
        "asr_model": "fake",
        "voice_models": "unused",
        "references": 1,
    }


class FakeTranscriber:
    def __call__(self, pcm16k, key=None):
        words = ["hôm", "nay", "giá", "xăng", "có", "tăng", "hai", "nghìn", "đồng", "rồi"]
        return [
            {"word": w, "start": 0.5 * i + 0.3, "end": 0.5 * i + 0.75} for i, w in enumerate(words)
        ]


def fake_call(env, script, request):
    from vn_av_df.generation.voice import audio as A

    t = np.arange(int(1.2 * 24000)) / 24000
    wave = 0.2 * np.sin(2 * np.pi * 180 * t).astype(np.float32)
    if request["mode"] == "inpaint":
        wave = 0.2 * np.sin(2 * np.pi * 150 * np.arange(int(6 * 24000)) / 24000).astype(np.float32)
    A.write_wav(request["out"], A.resample(wave, 24000), 48000)
    return {"out": request["out"]}


def fake_synthesize(cfg, video, audio, output, frames, width, height, **_):
    from vn_av_df.data.media import decode
    from vn_av_df.data.render import write_lossless

    media = decode(video, max_side=320, sample_rate=48000)
    fake = (media["frames"][:frames] * 0.6).astype(np.uint8)
    write_lossless(fake, media["pcm"], output)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Dựng một lần cho cả module: part sạch → plan → voice → lipsync → finalize."""
    import sys

    from fake_wandb import FakeWandb

    from vn_av_df.generation.generate import finalize, run_generation
    from vn_av_df.generation.plan import make_plan
    from vn_av_df.generation.voice.engines import Engine

    fake = FakeWandb()
    previous = sys.modules.get("wandb")
    sys.modules["wandb"] = fake  # W&B giả: kiểm mọi chỗ ghi log chạy được, không cần mạng
    tmp = tmp_path_factory.mktemp("bench")
    clean = tmp / "clean"
    clean_rows = make_clean(clean)
    datasets = tmp / "datasets"
    out = datasets / "vn-test-part1"
    cfg = {
        "root": str(tmp),
        "clean_dataset": str(clean),
        "generated_dataset": str(out),
        "plan": str(tmp / "plan.json"),
        "generation": options(),
        "device": "cpu",
        "dataset": str(datasets),
        "dataset_parts": ["vn-test-part1"],
        "cache": str(tmp / "cache"),
        "runs": str(tmp / "runs"),
        "seeds": [42],
        "wandb": {"project": "test", "mode": "offline"},
        "model_assets": {"xlsr": TINY_XLSR},
        "training": {
            "epochs": 1,
            "max_epochs": 1,
            "patience": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "hidden": 16,
            "dropout": 0.0,
            "batch_size": 4,
            "eval_batch_size": 4,
            "workers": 0,
        },
    }
    summary = make_plan(clean, cfg["plan"], cfg["generation"], seed=7)
    engine = Engine(cfg, call=fake_call)
    voice = run_generation(cfg, "voice", engine=engine, transcriber=FakeTranscriber())
    synth = {g: fake_synthesize for g in ("g_a", "g_b", "g_c", "g_new")}
    lip = run_generation(cfg, "lipsync", synthesizers=synth)
    info = finalize(cfg)
    yield {
        "wandb": fake,
        "cfg": cfg,
        "summary": summary,
        "voice": voice,
        "lip": lip,
        "info": info,
        "clean_rows": clean_rows,
        "out": out,
    }
    if previous is None:
        sys.modules.pop("wandb", None)
    else:
        sys.modules["wandb"] = previous


def test_cell_labels_follow_contract():
    rows = []
    for cell in ("real", "sham", "voice", "source", "donor", "both"):
        for kind in ("full", "partial"):
            if cell == "real" and kind == "partial":
                continue
            row = {
                "sample_id": f"{cell}_{kind}",
                "split": "train",
                "review_status": "keep",
                "duration_s": 6.0,
                "source_id": "src",
                "generator": "none",
                "generator_version": "x",
                "sha256": f"{len(rows):064d}",
                **labels(cell, kind, 6.0, (1.0, 2.5)),
            }
            if cell == "sham":
                row["control_type"] = "conventional_audio_splice"
            rows.append(row)
    validate_rows(rows)
    both = labels("both", "full", 6.0, (1.0, 2.5))
    assert both["visual_fake_intervals"] == [[0.0, 6.0]] and both["audio_fake_intervals"] == [
        [1.0, 2.5]
    ]
    assert labels("voice", "partial", 6.0, (1.0, 2.5))["av_mismatch_intervals"] == [[1.0, 2.5]]
    assert labels("sham", "full", 6.0)["label"] == 0


def test_plan_rotates_cells_generators_and_voices(world):
    summary = world["summary"]
    assert summary["parents"] == 18
    cells = summary["cells"]
    assert (
        set(cells) == {"source", "donor", "both"} and max(cells.values()) - min(cells.values()) <= 6
    )
    plan = json.loads(Path(world["cfg"]["plan"]).read_text(encoding="utf-8"))
    pairs = Counter(
        (j["parent"]["clip_id"], j["generator"]) for j in plan["jobs"] if j["type"] == "lipsync"
    )
    assert set(pairs.values()) == {1}
    test_gens = {
        j["generator"]
        for j in plan["jobs"]
        if j["type"] == "lipsync" and j["parent"]["split"] == "test"
    }
    assert "g_new" in test_gens
    train_voices = {
        j["voice"] for j in plan["jobs"] if j["type"] == "voice" and j["parent"]["split"] == "train"
    }
    assert train_voices <= {"vieneu", "f5_vi"}


def test_generation_finalize_counts_and_labels(world):
    info, out = world["info"], world["out"]
    assert world["voice"]["remaining"] == 0 and world["lip"]["remaining"] == 0
    rows = [
        json.loads(x) for x in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validate_rows(rows, out, verify_media=True)
    assert set(info["cells"]) == {"real", "sham", "voice", "source", "donor", "both"}

    def video_md5(row):
        path = out / row["video"]
        return run(
            [ffmpeg(), "-v", "error", "-i", path, "-map", "0:v", "-c", "copy", "-f", "md5", "-"]
        )

    # Job voice encode hình một lần, các sample còn lại chép luồng hình của real.
    voiced = next(r for r in rows if r["variant"] == "voice_full")
    real = next(
        r for r in rows if r["sample_id"] == voiced["sample_id"][: -len("voice_full")] + "real"
    )
    assert video_md5(voiced) == video_md5(real)
    test_voice = [r for r in rows if r["split"] == "test" and r["cell"] == "voice"]
    assert any(r["variant"] == "voice_inpaint" for r in test_voice)
    assert not any(r["variant"] == "voice_inpaint" for r in rows if r["split"] == "train")
    both = [r for r in rows if r["cell"] == "both" and r["kind"] == "partial"]
    assert both and all(r["visual_fake_intervals"] == r["audio_fake_intervals"] for r in both)
    audit = json.loads((out / "shortcut_audit.json").read_text(encoding="utf-8"))
    assert audit["samples"] == len(rows) and "lead_silence_s" in audit["overall"]
    assert "voice/full" in audit["by_cell_vs_real"]
    sham = [r for r in rows if r["cell"] == "sham"]
    assert all(r["label"] == 0 and r["av_mismatch_intervals"] for r in sham)
    # Tiếng sine liền không có chỗ nghỉ: donor cắt cứng và ghi lại.
    assert all(r["donor_fit"] == "cut" for r in rows if r["cell"] in ("sham", "donor"))
    sessions = json.loads((out / "generation.json").read_text(encoding="utf-8"))["sessions"]
    assert all("failed" in s and "environment" in s for s in sessions)
    assert {s["phase"] for s in sessions} == {"voice", "lipsync"}
    per_parent = Counter(r["source_clip_id"] for r in rows if r["split"] == "train")
    # train: real 1 + sham 2 + voice 2 + 3 generator × 2 = 11
    assert set(per_parent.values()) == {11}


def test_donor_audio_picks_best_fitting_donor(tmp_path):
    """Plan có donor_fit: thử mọi donor, lấy donor vừa nhất; plan cũ: cắt cứng như trước."""
    from types import SimpleNamespace

    from vn_av_df.data.media import decode
    from vn_av_df.generation.generate import donor_audio
    from vn_av_df.generation.plan import donors_for
    from vn_av_df.generation.voice import audio as A

    def clip(name, seconds, on):
        path = tmp_path / f"{name}.mkv"
        speech = f"0.3*sin(2*PI*200*t)*lt(mod(t\\,1)\\,{on})"
        run(
            [
                ffmpeg(),
                *("-nostdin", "-v", "error", "-y", "-f", "lavfi"),
                *("-i", f"testsrc2=s=64x48:r=25:d={seconds}"),
                *("-f", "lavfi", "-i", f"aevalsrc={speech}:s=48000:d={seconds}"),
                *("-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", path),
            ]
        )
        return {
            "clip_id": name,
            "video": path.name,
            "source_id": f"yt_{name}",
            "duration_s": seconds,
        }

    pool = [clip("a_steady", 4.0, 1.01), clip("b_gaps", 3.6, 0.7)]
    parent = np.zeros(76 * 1920, np.float32)
    parent[4800:-14400] = 0.3
    ctx = SimpleNamespace(
        clean=tmp_path,
        options={"donor_fit": {"min_pause_s": 0.02}},
        decode=lambda row: decode(tmp_path / row["video"], max_side=64, sample_rate=48000),
    )
    donor, pcm, note = donor_audio(ctx, {"donor_pool": pool}, parent, "sham_donor")
    assert donor["clip_id"] == "b_gaps" and len(pcm) == len(parent) and "donor_window_s" in note
    assert np.array(A.edge_silence(pcm)) / 48000 == pytest.approx([0.1, 0.24], abs=0.021)
    donor, pcm, note = donor_audio(ctx, {"sham_donor": pool[0]}, parent, "sham_donor")
    assert donor is pool[0] and note == {} and len(pcm) == len(parent)
    # Clip dài nhất của một người vẫn có donor ngắn hơn trong shorter_s.
    rows = [
        {"clip_id": c, "split": "train", "speaker_id": "s", "duration_s": d, "source_id": c}
        for c, d in (("x", 6.0), ("y", 5.5), ("z", 4.0))
    ]
    assert donors_for(rows[0], rows) == []
    assert [d["clip_id"] for d in donors_for(rows[0], rows, 1.0)] == ["y"]


def test_batches_split_clips_across_sessions(world, tmp_path, monkeypatch):
    """4 đợt phủ đủ clip; phiên voice đợt 1 ở máy A, máy B gộp rồi chỉ lip-sync đợt 1 trên 2 GPU."""
    import shutil
    import threading
    import time

    from vn_av_df.generation.generate import batch_clips, merge_previous, run_generation
    from vn_av_df.generation.voice.engines import Engine

    plan = json.loads(Path(world["cfg"]["plan"]).read_text(encoding="utf-8"))
    parts = [batch_clips(plan, (k, 4)) for k in range(1, 5)]
    clips = {j["parent"]["clip_id"] for j in plan["jobs"]}
    assert set().union(*parts) == clips and sum(map(len, parts)) == len(clips)
    with pytest.raises(ValueError):
        batch_clips(plan, (5, 4))

    def session(name):
        root = tmp_path / name
        root.mkdir()
        return {
            **world["cfg"],
            "root": str(root),
            "plan": str(root / "plan.json"),
            "generated_dataset": str(root / "datasets/vn-test-part1"),
        }

    a, b = session("a"), session("b")
    shutil.copy2(world["cfg"]["plan"], a["plan"])
    a["session_batch"] = (1, 4)
    run_generation(a, "voice", engine=Engine(a, call=fake_call), transcriber=FakeTranscriber())
    merge_previous(b, [a["root"]])
    synth = {g: fake_synthesize for g in ("g_a", "g_b", "g_c", "g_new")}
    with pytest.raises(RuntimeError, match="record giọng"):
        run_generation(b, "lipsync", synthesizers=synth)  # đợt 2-4 chưa có giọng
    b["session_batch"], b["generation_gpus"] = (1, 4), [0, 1]
    guard, busy, clash, events = threading.Lock(), set(), [], []

    def one_job_per_gpu(name):
        def synthesize(*args, gpu, **kw):
            with guard:
                clash.append(gpu in busy)
                busy.add(gpu)
                events.append((name, "start", len(busy)))
            try:
                time.sleep(0.05)
                fake_synthesize(*args, **kw)
            finally:
                with guard:
                    busy.discard(gpu)
                    events.append((name, "end", len(busy)))

        return synthesize

    result = run_generation(b, "lipsync", synthesizers={g: one_job_per_gpu(g) for g in synth})
    assert len(clash) > 2 and not any(clash)
    # Job đầu mỗi generator chạy một mình, các job sau song song trên 2 GPU.
    for g in {e[0] for e in events}:
        mine = [e[1] for e in events if e[0] == g]
        assert mine[:2] == ["start", "end"]
    assert max(e[2] for e in events) == 2
    keys = {j["key"] for j in plan["jobs"] if j["parent"]["clip_id"] in parts[0]}
    records = Path(b["generated_dataset"]) / "records"
    assert result["remaining"] == 0 and {p.stem for p in records.glob("*.json")} == keys
    assert not any("cell_fallback" in p.read_text(encoding="utf-8") for p in records.glob("*.json"))

    # Một generator lỗi (không phải lỗi bỏ qua được): chỉ nó dừng, generator khác vẫn xong.
    c = session("c")
    merge_previous(c, [a["root"]])
    c["session_batch"] = (1, 4)

    def broken(*args, **kw):
        raise RuntimeError("boom")

    seen = []

    def noting(name):
        return lambda *args, **kw: (seen.append(name), fake_synthesize(*args, **kw))

    c["session_generators"] = order = ["g_new", "g_c", "g_a", "g_b"]
    result = run_generation(
        c, "lipsync", synthesizers={**{g: noting(g) for g in synth}, "g_a": broken}
    )
    assert set(result["failed"]) == {"g_a"} and "boom" in result["failed"]["g_a"]
    assert list(dict.fromkeys(seen)) == [g for g in order if g in seen] and len(set(seen)) > 1
    done = {p.stem for p in (Path(c["generated_dataset"]) / "records").glob("*.json")}
    lip = [j for j in plan["jobs"] if j["type"] == "lipsync" and j["key"] in keys]
    assert {j["key"] for j in lip if j["generator"] != "g_a"} <= done
    assert not done & {j["key"] for j in lip if j["generator"] == "g_a"}

    # Job bỏ vì lỗi generator được phiên sau làm lại; gộp hai phiên theo thứ tự nào cũng lấy bản xong.
    d, e = session("d"), session("e")
    merge_previous(d, [a["root"]])
    d["session_batch"] = e["session_batch"] = (1, 4)

    def flaky(*args, **kw):
        raise RuntimeError("Worker failed: CUDA out of memory")

    run_generation(d, "lipsync", synthesizers={**synth, "g_a": flaky})
    retried = {j["key"] for j in lip if j["generator"] == "g_a"} & {
        p.stem for p in (Path(d["generated_dataset"]) / "records").glob("*.json")
    }
    assert retried
    # Setup của g_a đổi sau phiên lỗi (sửa code adapter): chưa xong job nào thì không chặn.
    log = Path(d["generated_dataset"]) / "generation.json"
    stale = json.loads(log.read_text(encoding="utf-8"))
    stale["provenance"]["g_a"] = {"version": "before-fix", "generator": "g_a"}
    log.write_text(json.dumps(stale), encoding="utf-8")
    merge_previous(e, [d["root"]])
    assert run_generation(e, "lipsync", synthesizers=synth)["remaining"] == 0
    merge_previous(e, [d["root"]])  # bản bỏ của d không đè bản đã làm của e
    merge_previous(d, [e["root"]])
    for key in retried:
        mine, theirs = (Path(s["generated_dataset"]) for s in (d, e))
        record, other = (
            json.loads((side / "records" / f"{key}.json").read_text(encoding="utf-8"))
            for side in (mine, theirs)
        )
        assert record == other and "rows" in record
        assert all(sha(mine / row["video"]) == row["sha256"] for row in record["rows"])
    assert json.loads(log.read_text(encoding="utf-8"))["provenance"]["g_a"]["version"] == "test"

    # Setup khác của generator đã có job xong thì dừng, trừ adapter cũ ghi trong REDO: làm lại.
    from vn_av_df.generation import generate

    log = Path(e["generated_dataset"]) / "generation.json"
    old = json.loads(log.read_text(encoding="utf-8"))
    old["provenance"]["g_b"] = {"adapter_sha256": "old", "generator": "g_b"}
    log.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(ValueError, match="g_b differs"):
        run_generation(e, "lipsync", synthesizers=synth)
    monkeypatch.setattr(generate, "REDO", {"g_b": "old"})
    seen.clear()
    result = run_generation(e, "lipsync", synthesizers={g: noting(g) for g in synth})
    assert result["remaining"] == 0 and result["done"] > 0 and set(seen) == {"g_b"}

    # Cả một người không thấy mặt (IP-LAP/mediapipe): bỏ hẳn từng job, không dừng generator.
    f = session("f")
    merge_previous(f, [a["root"]])
    f["session_batch"] = (1, 4)
    faceless = [j["key"] for j in lip if j["generator"] == "g_a"]
    assert faceless
    monkeypatch.setattr(generate, "MAX_SKIP_STREAK", len(faceless))

    def no_face(*args, **kw):
        raise RuntimeError("Worker failed: g_a error: raise NotImplementedError  # not detect face")

    result = run_generation(f, "lipsync", synthesizers={**synth, "g_a": no_face})
    assert "g_a" not in result["failed"] and result["remaining"] == 0
    records = Path(f["generated_dataset"]) / "records"
    kinds = {
        json.loads((records / f"{k}.json").read_text(encoding="utf-8"))["kind"] for k in faceless
    }
    assert kinds == {"face"}


def write_fake_caches(cfg, rows, dim=32):
    rng = np.random.default_rng(0)
    cache = Path(cfg["cache"])
    for row in rows:
        n = int(round(row["duration_s"] * 25))
        sid = row["sample_id"]
        a = rng.normal(size=(n, dim)).astype(np.float32)
        v = a + rng.normal(scale=0.5, size=(n, dim)).astype(np.float32)
        if row.get("av_mismatch_intervals"):
            v = rng.normal(size=(n, dim)).astype(np.float32)
        landmarks = rng.normal(100, 2, size=(n, 68, 2)).astype(np.float16)
        np.savez(
            cache / "avhubert" / f"{sid}.npz",
            audio=a,
            visual=v,
            audio_valid=np.ones(n, bool),
            visual_valid=np.ones(n, bool),
            times_s=np.arange(n) / 25,
            mouth_boxes=np.tile([[160, 160, 60]], (n, 1)).astype(np.float32),
            landmarks=landmarks,
        )
        np.savez(
            cache / "xlsr" / f"{sid}.npz",
            ssl=rng.normal(size=(n, 3, 1024)).astype(np.float16),
            times_s=np.arange(n) / 25,
            valid=np.ones(n, bool),
        )
        mouth = rng.integers(0, 255, size=(n, 96, 96)).astype(np.uint8)
        lips = rng.normal(scale=0.1, size=(n, 20, 2)).astype(np.float16)
        np.savez(
            cache / "mouth" / f"{sid}.npz",
            mouth=mouth,
            lips=lips,
            pose=rng.normal(scale=0.1, size=(n, 3)).astype(np.float32),
            valid=np.ones(n, bool),
            times_s=np.arange(n) / 25,
        )
        np.savez(
            cache / "lipinc" / f"{sid}.npz",
            frames=rng.integers(0, 255, size=(8, 64, 144, 3)).astype(np.uint8),
            frame_ids=np.arange(8, dtype=np.int32),
            valid=np.ones(n, bool),
            times_s=np.arange(n) / 25,
        )
        wave = rng.normal(scale=3000, size=(n, 640))
        if row.get("synthetic_audio"):
            wave = wave * 2
        np.savez(
            cache / "wave" / f"{sid}.npz",
            wave=wave.astype(np.int16),
            valid=np.ones(n, bool),
            times_s=np.arange(n) / 25,
        )


def test_every_model_forward_backward():
    torch.manual_seed(0)
    B, T, D = 2, 110, 32
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 100:] = False
    batch = {
        "a": torch.randn(B, T, D),
        "v": torch.randn(B, T, D),
        "ssl": torch.randn(B, T, 3, 1024),
        "mouth": torch.randint(0, 255, (B, T, 96, 96), dtype=torch.uint8),
        "lips": torch.randn(B, T, 20, 2) * 0.1,
        "pose": torch.randn(B, T, 3) * 0.1,
        "wave": torch.randn(B, T, 640) * 0.1,
        "lipinc": torch.randint(0, 255, (B, 8, 64, 144, 3), dtype=torch.uint8),
        "speaker": torch.tensor([0, 1]),
        "mask": mask,
        "y_ai": torch.zeros(B, T),
        "y_visual": torch.zeros(B, T),
        "y_audio": torch.zeros(B, T),
        "y_mismatch": torch.zeros(B, T),
        "index": torch.arange(B),
        "video": {
            "ai": torch.tensor([0.0, 1.0]),
            "visual": torch.tensor([0.0, 1.0]),
            "audio": torch.tensor([0.0, 1.0]),
            "mismatch": torch.tensor([0.0, 1.0]),
        },
    }
    for key in ("y_ai", "y_visual", "y_audio", "y_mismatch"):
        batch[key][1, 10:30] = 1
    batch["refs"] = {
        k: batch[k][:, None].repeat(1, 1, *([1] * (batch[k].dim() - 1)))[:, :, :40]
        for k in ("a", "v", "ssl", "mouth", "lips", "pose")
    }
    batch["refs"]["valid"] = torch.ones(B, 1, 40, dtype=torch.bool)
    batch["ref_count"] = torch.tensor([1, 1])
    for name in zoo.SPECS:
        model = zoo.build(name, dim=D, hidden=16, assets={"xlsr": TINY_XLSR, "n_speakers": 3})
        out = model(batch)
        for head in model.heads:
            assert out[head].shape == (B, T), name
            assert torch.isfinite(out[head][mask]).all(), name
        if model.trainable:
            loss = model.loss(batch, out)
            assert torch.isfinite(loss), name
            loss.backward()


def test_train_evaluate_report_smoke(world):
    from vn_av_df.dataset import training_dataset
    from vn_av_df.training.evaluate import evaluate_all
    from vn_av_df.training.prepare import encoders_for
    from vn_av_df.training.report import build_report
    from vn_av_df.training.train import train_all

    cfg = copy.deepcopy(world["cfg"])
    cfg["models"] = list(zoo.SPECS)  # 9 cấu hình của đội hình + speechforensics (demo)
    assert encoders_for(cfg["models"]) == [
        "landmarks",
        "avhubert",
        "mouth",
        "lipinc",
        "xlsr",
        "wave",
    ]
    for name in encoders_for(cfg["models"]):
        (Path(cfg["cache"]) / name).mkdir(parents=True, exist_ok=True)
    rows, _ = training_dataset(cfg)
    write_fake_caches(cfg, rows)
    report = train_all(cfg)
    assert set(report) == {f"{m}_seed42" for m in cfg["models"]}
    results = evaluate_all(cfg)
    assert results["avh_align_plus_seed42"]["mismatch"] is not None
    assert set(results["trihead_v2_seed42"]) >= {"ai", "visual", "audio", "mismatch"}
    assert set(results["bl_ens_seed42"]) == {
        "ai",
        "visual",
        "audio",
    }  # max(BL-V, BL-A), không train
    lipinc = json.loads(
        (Path(cfg["runs"]) / "lipinc_v2_seed42/metrics.json").read_text(encoding="utf-8")
    )
    assert not lipinc["frame_level"] and lipinc["tasks"]["visual"]["temporal"] is None
    history = json.loads(
        (Path(cfg["runs"]) / "avh_align_seed42/history.json").read_text(encoding="utf-8")
    )
    assert history["train_samples"] == sum(
        r["split"] == "train" and r.get("cell") == "real" for r in rows
    )  # BL-M chỉ học real
    metrics = json.loads(
        (Path(cfg["runs"]) / "trihead_v2_seed42/metrics.json").read_text(encoding="utf-8")
    )
    ai = metrics["tasks"]["ai"]
    assert ai["ref_norm"]["samples"] > 0 and "cell/both" in ai["breakdown"]
    assert ai["breakdown"]["generator_seen/False"]["n"] > 0
    assert {"gender/F", "gender/M"} <= ai["breakdown"].keys()  # phân rã theo giới
    point = ai["operating_points"]["fpr_0.05"]
    assert 0 <= point["tpr"] <= 1 and set(point["base_rates"]) == {"0.5", "0.1", "0.01"}
    assert "real" in ai["fpr_by_cell"]
    assert set(ai["temporal"]) >= {"ap@0.5", "ap@0.75", "ap@0.95", "ar@1", "ar@5", "ar@10"}
    with pytest.raises(ValueError, match="already run"):
        torch.save({"x": 1}, Path(cfg["runs"]) / "avh_align_plus_seed42/best.pt")
        evaluate_all(cfg)
    built = build_report(cfg)
    text = Path(built["report"]).read_text(encoding="utf-8")
    assert "| trihead_v2 |" in text and "| bl_ens |" in text and "Cổng" in text
    assert "Điểm vận hành" in text
    assert "## Cách đọc" in text and "bootstrap theo speaker" in text and "(chạy 1 seed)" in text
    gates = {(g["first"], g["second"], g["task"]): g for g in built["gates"]}
    assert ("trihead_v2", "bl_ens", "ai") in gates
    assert gates[("trihead_v2", "ssl_aasist", "audio")]["subset"] == "unseen_voice"
    # W&B chỉ ở quá trình train: không có run nào của bước sinh fake; run nào cũng đóng.
    fake = world["wandb"]
    assert all(r.finished for r in fake.runs)
    assert {r.kwargs["job_type"] for r in fake.runs} <= {
        "prepare",
        "train",
        "evaluate",
        "report",
        "benchmark",
    }
    # Tên hiển thị: "<YYYYmmdd-HHMM> <tên>" (giờ Việt Nam), id / tên giữ nguyên khi chạy tiếp.
    assert all(re.match(r"\d{8}-\d{4} ", r.kwargs["name"]) for r in fake.runs)
    trains = {r.kwargs["name"].split(" ", 1)[1]: r for r in fake.by_type("train")}
    m1 = trains["trihead_v2_seed42"]
    assert {"loss/train", "val_auc/mismatch", "grad_norm"} <= m1.keys()
    assert "best_epoch" in m1.summary and m1.artifacts[0].type == "model"
    assert m1.kwargs["config"]["augment_lag"] and m1.kwargs["config"]["pseudo_fake"] > 0
    assert not trains["avh_align_seed42"].kwargs["config"]["augment_lag"]
    assert trains["avh_align_seed42"].kwargs["config"]["real_only"]
    assert trains["ssl_aasist_seed42"].kwargs["config"]["training"]["lr"] == 1e-6  # công thức gốc
    evaluation = fake.by_type("evaluate")[-1]
    assert {"test/summary", "test/breakdown", "test/roc_ai"} <= evaluation.keys()
    assert {"report/gates", "report/auc_mean_std"} <= fake.by_type("report")[0].keys()


def test_resume_after_interruption_and_provenance(world, tmp_path, monkeypatch):
    """Ngắt giữa epoch 2 → chạy lại tiếp từ epoch 2 (không train lại epoch 1); last.pt bị xoá."""
    from vn_av_df.dataset import training_dataset
    from vn_av_df.training import train as T

    cfg = copy.deepcopy(world["cfg"])
    cfg["runs"] = str(tmp_path / "runs")
    cfg["models"], cfg["seeds"] = ["avh_align_plus"], [7]
    cfg["training"] = {**cfg["training"], "epochs": 3, "max_epochs": 3, "patience": 9}
    for name in ("avhubert", "mouth", "lipinc", "xlsr", "wave"):
        (Path(cfg["cache"]) / name).mkdir(parents=True, exist_ok=True)
    rows, _ = training_dataset(cfg)
    write_fake_caches(cfg, rows)
    calls, original = {"n": 0}, T.run_epoch

    def flaky(model, loader, optimizer, *args, **kwargs):
        if optimizer is not None:
            calls["n"] += 1
            if calls["n"] == 2:  # đang train epoch 2 thì "hết giờ phiên"
                raise KeyboardInterrupt
        return original(model, loader, optimizer, *args, **kwargs)

    monkeypatch.setattr(T, "run_epoch", flaky)
    out = Path(cfg["runs"]) / "avh_align_plus_seed7"
    with pytest.raises(KeyboardInterrupt):
        T.train_all(cfg)
    assert (out / "last.pt").is_file()
    assert not json.loads((out / "history.json").read_text(encoding="utf-8"))["complete"]
    monkeypatch.setattr(T, "run_epoch", original)
    T.train_all(cfg)
    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert history["complete"] and history["resumed_at"] == [2]
    assert [e["epoch"] for e in history["epochs"]] == [1, 2, 3]
    assert not (out / "last.pt").exists() and (out / "best.pt").is_file()
    records = json.loads((Path(cfg["runs"]) / "provenance.json").read_text(encoding="utf-8"))
    assert len(records) == 2  # mỗi lần chạy thêm một mục
    caches = records[-1]["caches"]
    assert set(caches) == {"landmarks", "avhubert", "mouth"}
    assert caches["avhubert"]["samples_found"] == 0  # cache giả không có .json chữ ký
    assert (
        records[-1]["selection"]["samples"] == len(rows)
        and "train.py" in records[-1]["code_sha256"]
    )
    assert records[-1]["environment"]["python"] and "hyperparameter_search" in records[-1]


def test_max_hours_stops_cleanly_then_resumes(world, tmp_path, monkeypatch):
    """Hết MAX_HOURS: dừng sau epoch (last.pt đã lưu, chưa xong), model sau không bắt đầu; phiên
    sau train tiếp đúng epoch. Grad norm NaN (AMP tràn số) không làm hỏng history.json."""
    from vn_av_df.dataset import training_dataset
    from vn_av_df.training import train as T

    cfg = copy.deepcopy(world["cfg"])
    cfg["runs"] = str(tmp_path / "runs")
    cfg["models"], cfg["seeds"] = ["avh_align_plus", "avh_align_sup"], [7]
    cfg["training"] = {**cfg["training"], "epochs": 3, "max_epochs": 3, "patience": 9}
    for name in ("avhubert", "mouth", "lipinc", "xlsr", "wave"):
        (Path(cfg["cache"]) / name).mkdir(parents=True, exist_ok=True)
    rows, _ = training_dataset(cfg)
    write_fake_caches(cfg, rows)
    clip = torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *a, **k: (clip(*a, **k), torch.tensor(float("nan")))[1],
    )
    cfg["deadline"] = time.time() + 1  # epoch sau chắc chắn không kịp
    report = T.train_all(cfg)
    out = Path(cfg["runs"]) / "avh_align_plus_seed7"
    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert not history["complete"] and history["stop"] == "time_limit"
    assert [e["epoch"] for e in history["epochs"]] == [1] and (out / "last.pt").is_file()
    parts = history["epochs"][0]["train_parts"]
    assert "grad_norm" not in parts and parts["grad_nonfinite_steps"] > 0
    assert report["avh_align_sup_seed7"]["stop"] == "time_limit"
    assert not (Path(cfg["runs"]) / "avh_align_sup_seed7").exists()
    assert T.missing_runs(cfg) == ["avh_align_plus_seed7", "avh_align_sup_seed7"]
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    cfg["deadline"] = None
    T.train_all(cfg)
    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert history["complete"] and history["resumed_at"] == [2]
    assert [e["epoch"] for e in history["epochs"]] == [1, 2, 3]
    assert T.missing_runs(cfg) == []


def test_run_lanes_trains_in_subprocesses_and_merges(world, tmp_path, capsys):
    """Hai lane (mỗi lane một tiến trình train_all, GPU riêng trên Kaggle): xong gộp về runs, tên
    nhóm W&B không đổi, thư mục lane được xoá."""
    from vn_av_df.dataset import training_dataset
    from vn_av_df.training.lanes import lane_dirs, run_lanes
    from vn_av_df.training.train import missing_runs

    cfg = copy.deepcopy(world["cfg"])
    cfg["runs"] = str(tmp_path / "runs" / "train_part1_v003")
    cfg["models"], cfg["seeds"] = ["avh_align_plus", "avh_align_sup"], [7]
    cfg["training"] = {**cfg["training"], "epochs": 1, "max_epochs": 1, "patience": 9}
    cfg["wandb"] = None
    for name in ("avhubert", "mouth", "lipinc", "xlsr", "wave"):
        (Path(cfg["cache"]) / name).mkdir(parents=True, exist_ok=True)
    rows, _ = training_dataset(cfg)
    write_fake_caches(cfg, rows)
    assert run_lanes(cfg, [["avh_align_plus"], ["avh_align_sup"]]) == [0, 0]
    out = capsys.readouterr().out
    assert "[GPU0]" in out and "[GPU1]" in out and "XONG" in out
    assert missing_runs(cfg) == []
    assert not any(d.exists() for d in lane_dirs(cfg, 2))
    records = json.loads((Path(cfg["runs"]) / "provenance.json").read_text(encoding="utf-8"))
    assert sorted(r["models"][0] for r in records) == ["avh_align_plus", "avh_align_sup"]


def test_prepare_retries_failed_samples(world, tmp_path, capsys):
    """Một mẫu lỗi (worker bị giết) không bỏ cả nhóm clip: thử lại ở cuối; vẫn lỗi thì báo rõ."""
    from vn_av_df.common.runtime import write_json
    from vn_av_df.training.prepare import prepare

    cfg = copy.deepcopy(world["cfg"])
    cfg["cache"], cfg["wandb"] = str(tmp_path / "cache"), None
    cfg["encoders"] = {"wave": {"kind": "wave", "workers_per_device": 2}}
    calls = Counter()

    class Flaky:
        def __init__(self, broken):
            self.broken = broken

        def extract(self, request, output):
            calls[output.name] += 1
            if output.name in self.broken and calls[output.name] <= self.broken[output.name]:
                raise RuntimeError("Worker exited (-9)")
            write_json(Path(output).with_suffix(".json"), {"ok": True})

    from vn_av_df.dataset import training_dataset

    sample = training_dataset(cfg)[0][0]["sample_id"] + ".npz"
    prepare(cfg, ["wave"], factory=lambda cfg, name, device: Flaky({sample: 1}))
    out = capsys.readouterr().out
    assert "thử lại 1 mẫu lỗi" in out and "đều được" in out and calls[sample] == 2
    info = json.loads((tmp_path / "cache/wave/extraction.json").read_text(encoding="utf-8"))
    assert info["environment"]["python"] and info["samples"] > 0
    calls.clear()
    with pytest.raises(RuntimeError, match="vẫn lỗi sau khi thử lại"):
        prepare(cfg, ["wave"], factory=lambda cfg, name, device: Flaky({sample: 9}))


def test_benchmark_estimates_epoch_time(world, tmp_path):
    from vn_av_df.dataset import training_dataset
    from vn_av_df.training.benchmark import benchmark

    cfg = copy.deepcopy(world["cfg"])
    cfg["runs"] = str(tmp_path / "runs")
    for name in ("avhubert", "mouth", "lipinc", "xlsr", "wave"):
        (Path(cfg["cache"]) / name).mkdir(parents=True, exist_ok=True)
    rows, _ = training_dataset(cfg)
    write_fake_caches(cfg, rows)
    report = benchmark(cfg, ["avh_align_plus", "trihead_v2_no_trace"], steps=1, warmup=0)
    for name in ("avh_align_plus", "trihead_v2_no_trace"):
        entry = report["models"][name]
        assert entry["error"] is None and entry["epoch_minutes"] > 0
        assert entry["max_hours_all_seeds"] == pytest.approx(
            entry["max_hours_per_seed"] * len(cfg["seeds"])
        )
    assert (Path(cfg["runs"]) / "benchmark.json").is_file()
