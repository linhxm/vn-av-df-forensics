from pathlib import Path

import numpy as np
import pytest
import torch

from vn_av_df.common.runtime import fingerprint, read_json, save_npz, sha, write_json
from vn_av_df.data.groups import write_manifest
from vn_av_df.dataset import SCHEMA
from vn_av_df.experiment import compare, evaluate, load_model, train_one
from vn_av_df.models import TemporalDetector


@pytest.mark.parametrize("architecture", ["linear", "gru", "tcn", "transformer"])
def test_masks_block_temporal_leakage_and_gradients(architecture):
    torch.manual_seed(2)
    model = TemporalDetector(8, 8, architecture, hidden=8, dropout=0).eval()
    a = torch.randn(12, 8)
    v = torch.randn(12, 8)
    valid = torch.ones(12, dtype=torch.bool)
    valid[5] = False
    one = model(a, v, valid)
    a2 = a.clone()
    a2[6:] += 100
    two = model(a2, v, valid)
    assert torch.allclose(one[:5], two[:5])
    one[valid].sum().backward()
    assert model.classifier.weight.grad.abs().sum() > 0
    assert torch.isfinite(model(a, v, torch.zeros_like(valid))).all()


@pytest.fixture
def project(tmp_path):
    rows = []
    cache = tmp_path / "cache"
    dataset = tmp_path / "dataset"
    rng = np.random.default_rng(4)
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            sid = f"{split}_{label}"
            digest = fingerprint(sid)
            row = {
                "sample_id": sid,
                "video": sid + ".mp4",
                "sha256": digest,
                "duration_s": 4.0,
                "label": label,
                "fake_intervals": [[1, 3]] if label else [],
                "split": split,
                "source_id": split,
                "speaker_id": split,
                "review_status": "keep",
                "generator": "wav2lip_gan" if label else "none",
                "generator_version": "fixture",
            }
            rows.append(row)
            path = cache / (sid + ".npz")
            save_npz(
                path,
                audio=rng.normal(size=(20, 8)).astype("float32"),
                visual=rng.normal(size=(20, 8)).astype("float32"),
                times_s=np.arange(20) * 0.2,
                audio_valid=np.ones(20, bool),
                visual_valid=np.ones(20, bool),
            )
            write_json(
                path.with_suffix(".json"),
                {
                    "feature_sha256": sha(path),
                    "feature_signature": "test-only",
                    "source_fingerprint": fingerprint(
                        {"assets": {"video": digest}, "variant": {"kind": "clean"}}
                    ),
                    "step_s": 0.2,
                    "window_s": 2.0,
                    "duration_s": 4.0,
                },
            )
    write_manifest(dataset / "manifest.jsonl", rows)
    write_json(
        dataset / "dataset_info.json",
        {
            "schema_version": SCHEMA,
            "samples": len(rows),
            "manifest_sha256": sha(dataset / "manifest.jsonl"),
        },
    )
    return {
        "dataset": str(dataset),
        "cache": str(cache),
        "runs": str(tmp_path / "runs"),
        "device": "cpu",
        "encoder": {},
        "architectures": ["gru", "tcn", "transformer"],
        "seeds": [42],
        "held_out_generators": [],
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "hidden": 8,
            "dropout": 0.0,
            "patience": 0,
            "top_fraction": 0.2,
        },
    }, rows


def test_compare_resume_test_thresholds_and_binary_output(project, tmp_path):
    cfg, rows = project
    report = compare(cfg)
    assert len(report["runs"]) == 3
    before = {r["checkpoint"]: load_model(r["checkpoint"])[1]["thresholds"] for r in report["runs"]}
    results = evaluate(cfg)
    assert len(results) == 3
    assert all("by_generator" in r for r in results)
    assert before == {p: load_model(p)[1]["thresholds"] for p in before}
    assert (Path(cfg["runs"]) / "test-summary.json").is_file()
    comparison_path = Path(cfg["runs"]) / "comparison.json"
    write_json(comparison_path, {**report, "selection": "changed after test"})
    with pytest.raises(ValueError, match="Test protocol changed"):
        evaluate(cfg)
    write_json(comparison_path, report)
    cfg["training"]["epochs"] = 2
    train_one(cfg, "gru", 42, resume=True)
    assert len(read_json(Path(cfg["runs"]) / "gru_seed42/history.json")) == 2
    from vn_av_df.inference import Analyzer

    class FixtureEncoder:
        signature = "test-only"

        def extract(self, row, output):
            source = Path(cfg["cache"]) / "test_0.npz"
            Path(output).write_bytes(source.read_bytes())
            return {"step_s": 0.2, "duration_s": 4.0}

    result = Analyzer(cfg, FixtureEncoder()).analyze("fixture.mp4")
    assert set(result) == {"video_score", "intervals"}
    assert 0 <= result["video_score"] <= 1
    cfg["training"]["hidden"] = 12
    with pytest.raises(ValueError, match="Resume"):
        train_one(cfg, "gru", 42, resume=True)


def test_fast_threshold_matches_exhaustive_search():
    from vn_av_df.metrics import binary_metrics, choose_threshold

    rng = np.random.default_rng(3)
    for _ in range(25):
        n = int(rng.integers(4, 300))
        labels = rng.integers(0, 2, n)
        labels[:2] = [0, 1]
        scores = np.round(rng.random(n), int(rng.integers(1, 4)))  # Có nhiều điểm hòa.
        candidates = np.unique(np.r_[0.5, scores, np.nextafter(scores, np.inf)])
        candidates = candidates[(candidates >= 0) & (candidates <= 1)]

        def key(t):
            m = binary_metrics(labels, scores, t)
            return m["f1"], -m["false_alarm_rate"]

        best = max(key(t) for t in candidates)
        chosen = choose_threshold(labels.tolist(), scores.tolist())
        assert key(chosen) == pytest.approx(best)
        assert chosen == max(t for t in candidates if key(t) == pytest.approx(best))


def test_resume_after_early_stop_keeps_selected_checkpoint(project):
    cfg, _ = project
    cfg["training"].update(epochs=2, patience=3)
    best = train_one(cfg, "gru", 42)
    state = torch.load(best.parent / "last.pt", weights_only=True)
    state["stale"] = 3  # Như thể phiên trước đã early stop.
    torch.save(state, best.parent / "last.pt")
    before = sha(best)
    cfg["training"]["epochs"] = 5
    train_one(cfg, "gru", 42, resume=True)
    assert len(read_json(best.parent / "history.json")) == 2
    assert sha(best) == before


def test_training_never_reads_test_cache(project):
    cfg, rows = project
    for r in rows:
        if r["split"] == "test":
            (Path(cfg["cache"]) / (r["sample_id"] + ".npz")).unlink()
    train_one(cfg, "gru", 42)


@pytest.mark.parametrize("methods", [["linear"], ["linear", "gru"], ["linear", "gru", "tcn"]])
def test_selected_training_reports_without_automatic_comparison(project, methods, monkeypatch):
    from vn_av_df.actions import execute
    from vn_av_df.inference import chosen_checkpoint

    cfg, rows = project
    cfg["architectures"] = methods
    execute("train", cfg)
    folder = Path(cfg["runs"])
    index = read_json(folder / "training.json")
    assert index["status"] == "complete"
    assert [r["architecture"] for r in index["runs"]] == methods
    assert not list(folder.glob("*comparison*"))
    assert not (folder / "test-lock.json").exists()
    for name in methods:
        detector = folder / f"{name}_seed42"
        predictions = read_json(detector / "predictions-validation.json")
        assert {p["sample_id"] for p in predictions} == {
            r["sample_id"] for r in rows if r["split"] == "validation"
        }
        assert all("label" in p and "fake_intervals" in p for p in predictions)
        assert read_json(detector / "best.json")["epoch"] == 1
    cfg["demo_method"] = methods[0]
    assert chosen_checkpoint(cfg) == folder / f"{methods[0]}_seed42/best.pt"
    if len(methods) == 1:
        monkeypatch.setenv("MPLBACKEND", "Agg")
        from vn_av_df.reporting import training_reports

        reports = training_reports(folder)
        assert len(reports) == 1 and reports[0]["summary"]["split"] == "validation"
        assert all(Path(p).read_bytes().startswith(b"\x89PNG") for p in reports[0]["images"])
        assert not (folder / "test-lock.json").exists()  # Rendering never calls test.
        evaluate(cfg)
        assert (folder / "evaluation.json").exists()
        assert not (folder / "test-comparison.json").exists()
        assert not (folder / "test-summary.json").exists()
        assert training_reports(folder, split="test")[0]["summary"]["split"] == "test"


def test_training_selection_cannot_silently_overwrite_run_index(project):
    from vn_av_df.experiment import train_selected

    cfg, _ = project
    cfg["architectures"] = ["linear"]
    train_selected(cfg)
    cfg["architectures"] = ["linear", "gru"]
    with pytest.raises(ValueError, match="selection changed"):
        train_selected(cfg, resume=True)


def test_train_part1_then_union_requires_new_run_and_records_parts(project):
    """An actual tiny CPU training run verifies selection reaches every model/test path."""
    cfg, rows = project
    from vn_av_df.dataset import training_dataset

    root = Path(cfg["dataset"])

    def part(name, items):
        folder = root / name
        write_manifest(folder / "manifest.jsonl", items)
        write_json(
            folder / "dataset_info.json",
            {
                "schema_version": SCHEMA,
                "samples": len(items),
                "manifest_sha256": sha(folder / "manifest.jsonl"),
            },
        )

    part("vn-av-df-data-part1", rows)
    cfg.update(dataset_parts=["vn-av-df-data-part1"], architectures=["gru"])
    compare(cfg)
    assert "vn-av-df-data-part1" in evaluate(cfg)[0]["by_part"]
    extra = []
    for r in rows:
        sid = "p2_" + r["sample_id"]
        digest = fingerprint(sid)
        extra.append(
            {
                **r,
                "sample_id": sid,
                "sha256": digest,
                "source_id": "p2_" + r["source_id"],
                "speaker_id": "p2_" + r["speaker_id"],
            }
        )
        cache = Path(cfg["cache"])
        target = cache / (sid + ".npz")
        target.write_bytes((cache / (r["sample_id"] + ".npz")).read_bytes())
        meta = read_json(cache / (r["sample_id"] + ".json"))
        meta["source_fingerprint"] = fingerprint(
            {"assets": {"video": digest}, "variant": {"kind": "clean"}}
        )
        write_json(target.with_suffix(".json"), meta)
    part("vn-av-df-data-part2", extra)
    cfg["dataset_parts"] = "all"
    assert len(training_dataset(cfg)[0]) == 12
    with pytest.raises(ValueError, match="Resume"):
        train_one(cfg, "gru", 42, resume=True)
    with pytest.raises(ValueError, match="Test protocol changed"):
        evaluate(cfg)
    cfg["runs"] += "-parts1-2"
    comparison = compare(cfg)
    assert len(comparison["dataset_selection"]["parts"]) == 2
    result = evaluate(cfg)[0]
    assert set(result["by_part"]) == {"vn-av-df-data-part1", "vn-av-df-data-part2"}
    assert read_json(Path(cfg["runs"]) / "dataset-selection.json")["samples"] == 12


def test_old_checkpoint_rejected(tmp_path):
    p = tmp_path / "old.pt"
    torch.save({"format": "fate-two-heads-v1"}, p)
    with pytest.raises(ValueError, match="incompatible"):
        load_model(p)


def test_artifact_checkpoint_relocated_from_kaggle(tmp_path):
    from vn_av_df.experiment import resolve_checkpoint

    p = tmp_path / "gru_seed42/best.pt"
    p.parent.mkdir()
    p.write_bytes(b"path-only")
    assert (
        resolve_checkpoint(
            {"runs": str(tmp_path)}, "/kaggle/working/project/runs/run/gru_seed42/best.pt"
        )
        == p
    )


def test_realrecon_native_training_freeze_resume_and_inference(project, monkeypatch):
    """Native25Hz→5Hz; stage A không đọc fake/test và luôn frozen ở stage B."""
    cfg, rows = project
    for r in rows:
        path = Path(cfg["cache"]) / (r["sample_id"] + ".npz")
        with np.load(path) as z:
            a = np.repeat(z["audio"], 5, axis=0)
            v = np.repeat(z["visual"], 5, axis=0)
        save_npz(
            path,
            audio=a,
            visual=v,
            times_s=np.arange(100) * 0.04,
            audio_valid=np.ones(100, bool),
            visual_valid=np.ones(100, bool),
        )
        meta = read_json(path.with_suffix(".json"))
        meta.update(step_s=0.04, output_stride=5, feature_sha256=sha(path))
        write_json(path.with_suffix(".json"), meta)
    cfg["architectures"] = ["realrecon"]
    cfg["reconstruction"] = {"epochs": 2, "lr": 0.001, "patience": 0}
    p = train_one(cfg, "realrecon", 42)
    model, state = load_model(p)
    assert state["reconstruction_report"]["train_ids"] == ["train_0"]
    assert state["reconstruction_report"]["validation_ids"] == ["validation_0"]
    before = {k: v.clone() for k, v in model.reconstructor.state_dict().items()}
    model.train()
    assert not model.reconstructor.training
    a, v = torch.randn(100, 8), torch.randn(100, 8)
    valid = torch.ones(100, dtype=torch.bool)
    valid[49] = False
    scores = model(a, v, valid)
    assert scores.shape == (20,)
    assert not model.output_valid(valid)[9]
    scores.sum().backward()
    assert all(p.grad is None for p in model.reconstructor.parameters())
    assert all(torch.equal(before[k], x) for k, x in model.reconstructor.state_dict().items())
    cfg["training"]["epochs"] = 2
    train_one(cfg, "realrecon", 42, resume=True)
    cfg["reconstruction"]["epochs"] = 3
    with pytest.raises(ValueError, match="Resume"):
        train_one(cfg, "realrecon", 42, resume=True)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from vn_av_df.reporting import detector_report

    report = detector_report(p.parent)
    assert report["summary"]["reconstruction"]["best_validation_loss"] >= 0
    assert any(Path(image).name == "reconstruction.png" for image in report["images"])


@pytest.mark.parametrize("architecture", ["realrecon", "realrecon_concat", "visual_tcn"])
def test_reconstruction_no_information_crosses_gap(architecture):
    from vn_av_df.models import build_model

    model = build_model(8, 8, architecture, 8, 0, native_stride=5).eval()
    model.reconstruction_ready.fill_(True)
    a, v = torch.randn(100, 8), torch.randn(100, 8)
    mask = torch.ones(100, dtype=torch.bool)
    mask[50:55] = False
    first = model(a, v, mask)
    a[55:] += 100
    v[55:] -= 100
    second = model(a, v, mask)
    assert torch.allclose(first[:10], second[:10])


def test_native_features_pooled_for_linear_but_not_reconstruction(tmp_path):
    from vn_av_df.features.registry import feature_arrays

    path = tmp_path / "cache.npz"
    save_npz(
        path,
        audio=np.ones((13, 4)),
        visual=np.ones((13, 6)),
        times_s=np.arange(13) * 0.04,
        audio_valid=np.array([True] * 12 + [False]),
        visual_valid=np.ones(13, bool),
    )
    arrays, times, meta = feature_arrays(path, {"step_s": 0.04, "output_stride": 5}, "linear")
    assert arrays[0].shape == (3, 4)
    assert not arrays[2][-1]
    assert meta["step_s"] == 0.2
    arrays, times, meta = feature_arrays(path, {"step_s": 0.04, "output_stride": 5}, "realrecon")
    assert arrays[0].shape == (13, 4) and len(times) == 3
