import numpy as np
import pytest

from vn_av_df.dataset import protocol_rows, temporal_targets, validate_rows
from vn_av_df.generation import compose_partial
from vn_av_df.metrics import choose_threshold, intervals, temporal_ap


def row(sid="a", split="train", label=0):
    return {
        "sample_id": sid,
        "split": split,
        "label": label,
        "duration_s": 4.0,
        "fake_intervals": [] if label == 0 else [[1, 2]],
        "review_status": "keep",
        "source_id": sid,
        "speaker_id": sid,
        "sha256": sid[0] * 64,
        "generator": "none" if label == 0 else "wav2lip_gan",
        "generator_version": "fixture",
    }


def test_transitive_donor_leakage():
    rows = [row("a"), row("b", "test", 1)]
    rows[1]["parent_ids"] = ["a"]
    with pytest.raises(ValueError, match="leakage"):
        validate_rows(rows)


def test_modality_union_and_reviewed_repost_leakage():
    r = row(label=1)
    r.update(
        synthetic_audio=False,
        synthetic_visual=True,
        audio_fake_intervals=[],
        visual_fake_intervals=[[1, 2]],
    )
    validate_rows([r])
    r["visual_fake_intervals"] = [[1, 3]]
    with pytest.raises(ValueError, match="Union"):
        validate_rows([r])
    copies = [row("a"), row("b", "test")]
    for item in copies:
        item["duplicate_group_id"] = "reviewed-repost"
    with pytest.raises(ValueError, match="leakage"):
        validate_rows(copies)


def test_paired_bootstrap_uses_same_groups_and_common_coverage():
    from vn_av_df.metrics import grouped_auc_interval, paired_auc_interval

    first, second = [], []
    for group in range(6):
        for label in (0, 1):
            item = row(f"g{group}_{label}", "test", label)
            item.update(
                source_id=f"source{group}",
                speaker_id=f"speaker{group}",
                sha256=f"{group * 2 + label:064x}",
            )
            first.append(dict(row=item, video_score=float(label)))
            second.append(dict(row=item, video_score=float(1 - label)))
    result = paired_auc_interval(first, second, replicates=50)
    assert result["groups"] == 6
    assert result["delta_auc"] == 1
    assert result["delta_auc_95ci"] == [1, 1]
    assert grouped_auc_interval(first[:4])["roc_auc_95ci"] is None
    second[0]["video_score"] = None
    assert paired_auc_interval(first, second, replicates=10)["samples"] == 11


def test_unknown_intervals_and_stress_controls():
    r = row(label=1)
    r["fake_intervals"] = None
    validate_rows([r])
    assert np.all(temporal_targets(r, np.arange(10) * 0.2, 0.2) == -1)
    r["generator"] = "global_lag"
    with pytest.raises(ValueError, match="controls"):
        validate_rows([r])
    r = row()
    r["fake_intervals"] = None
    with pytest.raises(ValueError, match="Real"):
        validate_rows([r])


def test_short_fake_and_unobserved_gap():
    r = row(label=1)
    r["fake_intervals"] = [[0.05, 0.1]]
    y = temporal_targets(r, np.arange(4) * 0.2, 0.2)
    assert y[0] == pytest.approx(0.25)
    result = intervals(np.ones(4), np.array([1, 0, 1, 1], bool), np.arange(4) * 0.2, 0.2, 0.7, 0.5)
    assert [(p["start_sec"], p["end_sec"]) for p in result] == [(0, 0.2), (0.4, 0.7)]


def test_temporal_ap_penalizes_duplicates_and_false_alarms():
    truth = {"real": [], "fake": [[1, 2], [3, 4]]}
    pred = {
        "real": [{"start_sec": 0, "end_sec": 1, "score": 0.99}],
        "fake": [
            {"start_sec": 1, "end_sec": 2, "score": 0.9},
            {"start_sec": 1, "end_sec": 2, "score": 0.8},
            {"start_sec": 3, "end_sec": 4, "score": 0.7},
        ],
    }
    result = temporal_ap(pred, truth, 0.5)
    assert result["matched"] == 2
    assert result["ap"] == pytest.approx(0.5)
    assert result["start_mae_s"] == 0
    assert temporal_ap({}, truth, 0.5)["ap"] == 0


def test_partial_pixels_and_pcm_exact_bounds():
    real = {"frames": np.zeros((100, 4, 4, 3), np.uint8), "pcm": np.zeros(192000, np.float32)}
    fake = {"frames": np.ones_like(real["frames"]), "pcm": np.ones_like(real["pcm"])}
    frames, pcm, spans = compose_partial(real, fake, (20, 30))
    assert spans == [[0.8, 1.2]]
    assert frames.sum() == 10 * 4 * 4 * 3
    assert pcm.sum() == 10 * 1920
    assert real["frames"].sum() == 0


def test_generator_holdout_not_in_validation():
    rows = [row("a", s, 1) for s in ("train", "validation", "test")]
    result = protocol_rows(rows, ["wav2lip_gan"])
    assert len(result) == 1 and result[0]["split"] == "test"
    assert 0 <= choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) <= 1
