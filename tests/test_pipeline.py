"""Benchmark v003 end-to-end trên dữ liệu tổng hợp (ASR / TTS / generator giả lập, CPU)."""

import copy
import json
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
    per_parent = Counter(r["source_clip_id"] for r in rows if r["split"] == "train")
    # train: real 1 + sham 2 + voice 2 + 3 generator × 2 = 11
    assert set(per_parent.values()) == {11}


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
    assert encoders_for(cfg["models"]) == ["avhubert", "mouth", "lipinc", "xlsr", "wave"]
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
    gates = {(g["first"], g["second"], g["task"]): g for g in built["gates"]}
    assert ("trihead_v2", "bl_ens", "ai") in gates
    assert gates[("trihead_v2", "ssl_aasist", "audio")]["subset"] == "unseen_voice"
    # W&B chỉ ở quá trình train: không có run nào của bước sinh fake; run nào cũng đóng.
    fake = world["wandb"]
    assert all(r.finished for r in fake.runs)
    assert {r.kwargs["job_type"] for r in fake.runs} <= {"prepare", "train", "evaluate", "report"}
    trains = {r.kwargs["name"]: r for r in fake.by_type("train")}
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
    assert set(caches) == {"avhubert", "mouth"}
    assert caches["avhubert"]["samples_found"] == 0  # cache giả không có .json chữ ký
    assert (
        records[-1]["selection"]["samples"] == len(rows)
        and "train.py" in records[-1]["code_sha256"]
    )


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
