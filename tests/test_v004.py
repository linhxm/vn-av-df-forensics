"""v004: quy ước độ trễ, nhãn mismatch đo được (0 / 1 / mask), loss có mask, M3 và BL-A2."""

import json

import numpy as np
import pytest
import torch

from vn_av_df import tracking
from vn_av_df.training import models as zoo
from vn_av_df.training import sync_labels as SL
from vn_av_df.training.data import frame_targets, lag_audio, video_targets


# ---------------------------------------------------------------- quy ước độ trễ (một cho mọi chỗ)
def test_lag_curve_and_at_shift_follow_lag_audio():
    """Dương = tiếng trễ ở tracking.lag_curve, data.lag_audio và sync_labels.at_shift."""
    rng = np.random.default_rng(0)
    n, d = 120, 16
    v = rng.normal(size=(n, d)).astype(np.float32)
    a = v.copy()
    base = tracking.lag_curve(a, v, np.ones(n, bool))
    assert int(np.nanargmax(base)) - SL.MAX_LAG == 0
    for k in (-2, 2, 3, 5):
        item = lag_audio({"a": a.copy()}, k)
        shifted = tracking.lag_curve(item["a"], v, item["lag_valid"])
        assert int(np.nanargmax(shifted)) - SL.MAX_LAG == k  # tiếng trễ k → đỉnh ở +k
        # độ tệ tại lag 0 sau trượt k = đường cong gốc tại lag −k
        expected = SL.at_shift(SL.badness("avhubert", base), k)
        assert -shifted[SL.MAX_LAG] == pytest.approx(expected, abs=1e-5)


def test_sync_v4_volume_uses_audio_delay_convention():
    torch.manual_seed(0)
    v = torch.randn(1, 80, 16)
    a = zoo.shift(v, -3)  # a_t = v_(t−3): tiếng trễ 3 frame
    mask = torch.ones(1, 80, dtype=torch.bool)
    trunk = zoo.SyncTrunkV4(16, hidden=8, phi=zoo.AlignMLP(16))
    volume, valid = trunk.volume(a, v, mask)
    assert volume.shape == (1, 3, 80, 31)  # chiếu học được, cos, Φ
    cosine = (volume[0, 1] * valid[0]).sum(0) / valid[0].sum(0).clamp(min=1)
    assert int(cosine.argmax()) - zoo.MAX_LAG == 3
    out = trunk(a, v, mask)
    assert set(out) == {"sync_rel", "sync_abs", "appearance", "global_lag"}
    assert out["sync_rel"].shape == (1, 80, 8) and out["global_lag"].shape == (1,)


# ---------------------------------------------------------------- luật nhãn đo được
def peaked(meter, at, rng, flat=False, better=False):
    """Đường cong tổng hợp: AV-HuBERT cos (đỉnh cao), SyncNet khoảng cách (đáy thấp). better:
    đỉnh rõ hơn real một chút (fake khớp tốt), để không rơi vào vùng xám p95-p99."""
    lags = np.arange(-SL.MAX_LAG, SL.MAX_LAG + 1)
    good, bad = (0.8, 0.2) if meter == "avhubert" else (6.0, 12.0)
    curve = np.full(len(lags), bad) + rng.normal(scale=0.01, size=len(lags))
    if not flat:
        margin = (0.05 if meter == "avhubert" else -0.5) if better else 0.0
        curve[lags == at] = good + margin + (0 if better else rng.normal(scale=0.01))
    return curve.tolist()


@pytest.fixture
def labeled(tmp_path):
    rng = np.random.default_rng(1)
    rows, curves = [], {"avhubert": {}, "syncnet": {}}

    def add(row, peak=0, flat=False, meters=("avhubert", "syncnet"), at=None):
        row = {"split": "train", "duration_s": 4.0, "av_mismatch_intervals": [], **row}
        row.setdefault("label", int(row["cell"] in SL.LIPSYNC_CELLS))
        rows.append(row)
        for meter in ("avhubert", "syncnet"):
            where = peak if at is None or meter not in at else at[meter]
            better = row["cell"] != "real"
            curves[meter][row["sample_id"]] = peaked(meter, where, rng, flat, better)

    for i in range(40):
        add({"sample_id": f"r{i}", "cell": "real", "source_clip_id": f"c{i}"})
    add({"sample_id": "rflat", "cell": "real", "source_clip_id": "cflat"}, flat=True)
    lipsync = {"cell": "source", "generator": "g", "visual_fake_intervals": [[0.0, 4.0]]}
    add({"sample_id": "late_full", "kind": "full", "source_clip_id": "c0", **lipsync}, peak=3)
    add(
        {
            "sample_id": "late_partial",
            "kind": "partial",
            "source_clip_id": "c0",
            **lipsync,
            "visual_fake_intervals": [[1.0, 2.0]],
        },
        peak=-9,  # số đo của chính bản partial không được dùng
    )
    add({"sample_id": "sync_full", "kind": "full", "source_clip_id": "c1", **lipsync}, peak=0)
    add(
        {"sample_id": "disagree_full", "kind": "full", "source_clip_id": "c2", **lipsync},
        at={"avhubert": 4, "syncnet": 0},
    )
    add({"sample_id": "noref_full", "kind": "full", "source_clip_id": "cflat", **lipsync}, peak=4)
    add({"sample_id": "flat_full", "kind": "full", "source_clip_id": "c3", **lipsync}, flat=True)
    add(
        {
            "sample_id": "sham_full",
            "cell": "sham",
            "source_clip_id": "c4",
            "av_mismatch_intervals": [[0.0, 4.0]],
        }
    )
    files = {}
    for meter, key in SL.CURVE_KEY.items():
        path = tmp_path / "m" / f"{meter}.json"
        path.parent.mkdir(exist_ok=True)
        samples = {sid: {key: c} for sid, c in curves[meter].items()}
        path.write_text(
            json.dumps({"convention": "index i <-> audio lag i - 15", "samples": samples})
        )
        files[meter] = str(path)
    cfg = {
        "dataset": str(tmp_path),
        "dataset_parts": None,
        "cache": str(tmp_path / "cache"),
        # sharpness_quantile thấp: real tổng hợp đều nhọn như nhau, chỉ rflat là không đo được
        # (mặc định p10 luôn coi ~10 % real là không đo được).
        "mismatch_labels": {
            "mode": "measured",
            "min_reals": 10,
            "sharpness_quantile": 0.02,
            "files": files,
        },
    }
    out = SL.annotate(cfg, rows)
    return {r["sample_id"]: r for r in out}, out


def test_measured_states_follow_rule(labeled):
    by_id, _ = labeled
    assert by_id["r5"]["_mismatch_eval"] == 0
    assert by_id["sham_full"]["_mismatch_eval"] == 1  # lệch nội dung: theo cách tạo
    assert by_id["late_full"]["_mismatch_eval"] == 1  # trễ 3 frame, hai thước đồng ý
    assert by_id["late_partial"]["_mismatch_eval"] == 1  # dùng số đo bản full cùng lần chạy
    assert by_id["sync_full"]["_mismatch_eval"] == 0
    assert by_id["disagree_full"]["_mismatch_eval"] is None  # hai thước không đồng ý → mask
    assert by_id["noref_full"]["_mismatch_eval"] is None  # clip real gốc không đo được → mask
    # đường cong phẳng mà tại lag 0 tệ (không có đỉnh) vẫn là lệch, miễn clip gốc đo được
    assert by_id["flat_full"]["_mismatch_eval"] == 1


def test_measured_states_change_with_shift(labeled):
    by_id, _ = labeled
    real, late, flat = by_id["r5"], by_id["late_full"], by_id["rflat"]
    assert SL.video_state(real, 1) == 0  # |k| ≤ near_shift
    assert SL.video_state(real, 5) == 1  # real bị trễ 5 frame: lệch, AI = 0
    assert SL.video_state(flat, 5) is None  # real không đo được: không biết
    assert SL.video_state(late, -3) == 0  # trượt bù đúng độ trễ của generator → khớp
    assert video_targets(real, 5)["mismatch"] == 1.0
    assert video_targets(flat, 5)["mismatch"] == -1.0


def test_frame_targets_mask_and_partial_span(labeled):
    by_id, _ = labeled
    partial = by_id["late_partial"]
    y, valid = SL.frame_targets(partial, 100, 0)
    assert valid.all() and y[25:50].min() == 1 and y[:20].max() == 0 and y[55:].max() == 0
    y, valid = SL.frame_targets(by_id["disagree_full"], 100, 0)
    assert not valid.any()
    targets = frame_targets(by_id["sham_full"], 100, 0)
    assert targets["mismatch"].min() == 1 and targets["mismatch_valid"].all()


def test_summary_counts_and_audit(labeled):
    _, rows = labeled
    out = SL.summary(rows)
    assert out["rule"] == SL.RULE_VERSION and set(out["thresholds"]) == {"avhubert", "syncnet"}
    assert out["counts"]["train/source/g"] == {"1": 3, "0": 1, "mask": 2}
    assert out["real_train_by_shift"]["0"]["0"] == 41
    assert out["lag0_audit_avhubert"]["train/g"]["flag"]


def test_construction_mode_keeps_v003_labels():
    row = {"sample_id": "x", "duration_s": 4.0, "av_mismatch_intervals": [], "cell": "source"}
    assert SL.annotate({"mismatch_labels": {"mode": "construction"}}, [row]) == [row]
    y, valid = SL.frame_targets(row, 50, 3)
    assert y.max() == 0 and valid.all()
    assert SL.video_state(row, 3) == 0


def test_missing_meter_fails_loudly(tmp_path):
    rows = [
        {"sample_id": f"r{i}", "split": "train", "cell": "real", "duration_s": 4.0}
        for i in range(3)
    ]
    cfg = {
        "dataset": str(tmp_path),
        "dataset_parts": None,
        "cache": str(tmp_path / "cache"),
        "mismatch_labels": {"mode": "measured"},
    }
    with pytest.raises(ValueError, match="sync-profile"):
        SL.annotate(cfg, rows)


# ---------------------------------------------------------------- loss có mask, M3, BL-A2
def small_batch(B=2, T=60, D=16, groups=3):
    torch.manual_seed(0)
    mask = torch.ones(B, T, dtype=torch.bool)
    batch = {
        "a": torch.randn(B, T, D),
        "v": torch.randn(B, T, D),
        "ssl": torch.randn(B, T, groups, 1024),
        "mouth": torch.randint(0, 255, (B, T, 96, 96), dtype=torch.uint8),
        "lips": torch.randn(B, T, 20, 2) * 0.1,
        "pose": torch.randn(B, T, 3) * 0.1,
        "prosody": torch.randn(B, T, 5),
        "head": torch.randn(B, T, 5),
        "cell_index": torch.tensor([4, 0]),
        "mask": mask,
        "speaker": torch.tensor([0, 1]),
        "index": torch.arange(B),
        "lag_shift": torch.tensor([3.0, 0.0]),
        "offset_valid": torch.tensor([True, False]),
        "y_mismatch_valid": torch.ones(B, T, dtype=torch.bool),
    }
    for key in ("y_ai", "y_visual", "y_audio", "y_mismatch"):
        batch[key] = torch.zeros(B, T)
    batch["y_mismatch"][1, 10:20] = 1
    return batch


def test_masked_mismatch_frames_do_not_change_loss():
    model = zoo.build("avh_align_plus", dim=16, hidden=8, assets={"load_pretrained": False})
    model.eval()
    batch = small_batch()
    batch["y_mismatch_valid"][0, 30:40] = False
    out = model(batch)
    first = float(model.loss(batch, out))
    batch["y_mismatch"][0, 30:40] = 1  # nhãn ở frame mask đổi → loss không đổi
    assert float(model.loss(batch, out)) == pytest.approx(first)
    batch["y_mismatch_valid"][0] = True  # bỏ mask → nhãn đó được tính
    assert float(model.loss(batch, out)) != pytest.approx(first)


@pytest.mark.parametrize(
    "name", ["trihead_v4", "trihead_v4_abs", "trihead_v4_no_phi", "trihead_v4_no_prosody"]
)
def test_trihead_v4_streams_offset_loss_and_groups(name):
    assets = {"load_pretrained": False, "n_speakers": 2, "ssl_groups": 5}
    model = zoo.build(name, dim=16, hidden=8, assets=assets)
    assert model.references == 0  # tham chiếu tắt mặc định
    sync = "sync_abs" if name.endswith("_abs") else "sync_rel"
    prosody = [] if name.endswith("_no_prosody") else ["prosody"]
    assert model.visual_streams == ["trace", sync, "appearance", *prosody]
    assert model.mismatch_streams == ["sync_abs", "audio", *prosody]
    assert ("prosody" in model.needs) == bool(prosody)
    assert (model.sync.phi is None) == name.endswith("_no_phi")
    assert model.rebuild_assets()["ssl_groups"] == 5
    batch = small_batch(groups=5)
    model.train()
    out = model(batch)
    loss = model.loss(batch, out)
    assert "offset" in model.loss_parts and torch.isfinite(loss)
    assert out["_cell"].shape == (2, 6) and "cell" in model.loss_parts  # đầu loại thao tác
    loss.backward()
    groups = model.optimizer_groups(1e-3)
    if model.sync.phi is not None:
        assert groups[0]["lr"] == model.phi_lr


def test_trihead_v4_ref_and_stream_drop():
    model = zoo.build(
        "trihead_v4_ref", dim=16, hidden=8, assets={"load_pretrained": False, "n_speakers": 2}
    )
    model.stream_drop = 1.0
    batch = small_batch()
    batch["refs"] = {
        k: batch[k][:, None, :30]
        for k in ("a", "v", "ssl", "mouth", "lips", "pose", "prosody", "head")
    }
    batch["refs"]["valid"] = torch.ones(2, 1, 30, dtype=torch.bool)
    batch["ref_count"] = torch.tensor([1, 1])
    model.train()
    out = model(batch)
    assert out["visual"].shape == (2, 60)
    model.loss(batch, out).backward()


def test_prosody_features_follow_pitch_and_voicing():
    from vn_av_df.training.prosody import prosody_features

    t = np.arange(16000 * 2) / 16000
    f0 = np.where(t < 1, 220.0, 120.0)  # 1 s giọng cao rồi 1 s giọng thấp
    wave = 0.3 * np.sin(2 * np.pi * np.cumsum(f0) / 16000)
    wave[: 16000 // 5] = 0.0  # 0,2 s im lặng đầu
    feats = prosody_features(wave.reshape(-1, 640))
    assert feats.shape == (50, 5)
    high, low, silent = feats[8:22], feats[28:48], feats[:4]
    assert high[:, 0].mean() > 0.5 > -0.5 > low[:, 0].mean()  # log F0 chuẩn hoá theo clip
    assert high[:, 1].mean() > 0.7 and silent[:, 1].max() == 0  # hữu thanh / vô thanh
    assert silent[:, 0].max() == 0  # F0 = 0 ở frame vô thanh


def test_head_features_eyebrow_raise_and_pose_velocity():
    from vn_av_df.training.prosody import coupling, head_features

    T = 60
    landmarks = np.zeros((T, 68, 2), np.float32)
    landmarks[:, 36:42, 0], landmarks[:, 42:48, 0] = 100, 160  # hai mắt cách 60 px
    landmarks[:, 36:48, 1] = 200
    brow = 170 - 10 * np.sin(np.linspace(0, 6, T))  # lông mày lên xuống
    landmarks[:, 17:27, 1] = brow[:, None]
    pose = np.zeros((T, 3), np.float32)
    pose[:, 1] = np.linspace(0, 0.2, T)  # gật đầu đều
    feats = head_features(landmarks, pose, np.ones(T, bool))
    assert feats.shape == (T, 5)
    assert np.corrcoef(feats[:, 0], 200 - brow)[0, 1] > 0.99  # độ nâng lông mày
    assert feats[1:, 3] == pytest.approx(np.full(T - 1, 0.2 / (T - 1) * 10), rel=1e-4)
    prosody = np.zeros((T, 5), np.float32)
    prosody[:, 3] = np.r_[0, np.diff(feats[:, 0])]  # ngữ điệu đi cùng lông mày → tương quan cao
    assert coupling(prosody, feats, np.ones(T, bool)) > 0.9


def test_audio_expert_and_group_guard():
    model = zoo.build("xlsr_frozen", dim=16, hidden=8, assets={"ssl_groups": 3})
    assert model.needs == {"ssl"} and not model.lag_augment
    batch = small_batch(groups=3)
    out = model(batch)
    assert out["audio"].shape == (2, 60)
    with pytest.raises(ValueError, match="nhóm lớp"):
        model(small_batch(groups=5))


def test_pos_weights_and_target_skip_masked():
    from vn_av_df.training.evaluate import known, target
    from vn_av_df.training.train import pos_weights

    rows = [
        {"label": 0, "_mismatch": {}, "_mismatch_eval": 0},
        {"label": 1, "_mismatch": {}, "_mismatch_eval": None},
        {"label": 1, "_mismatch": {}, "_mismatch_eval": 1},
        {"label": 0, "_mismatch": {}, "_mismatch_eval": 0},
    ]
    assert [target(r, "mismatch") for r in rows] == [0, None, 1, 0]
    assert len(known(rows, "mismatch")) == 3
    assert pos_weights(rows)["mismatch"] == pytest.approx(2.0)


# ---------------------------------------------------------------- chọn checkpoint, scheduler, log setup
def _rows(n=8):
    rows = []
    for i in range(n):
        fake = i % 2
        rows.append(
            {
                "sample_id": f"s{i}",
                "split": "validation",
                "cell": "voice" if fake else "real",
                "label": fake,
                "synthetic_visual": bool(fake and i % 4 == 1),
                "synthetic_audio": bool(fake),
                "av_mismatch_intervals": [[0, 1]] if fake else [],
                "duration_s": 1.0,
            }
        )
    return rows


def test_monitor_uses_auc_of_model_heads_and_falls_back_to_loss():
    from vn_av_df.training.train import better, monitor_spec, monitor_value

    rows = _rows()
    m3 = zoo.build("trihead_v4", 16, 8, 0.0, assets={"n_speakers": 0, "ssl_groups": 2})
    assert monitor_spec(m3, rows, {"monitor": "auc"}) == ("auc", ["audio", "mismatch", "visual"])
    assert monitor_spec(m3, rows, {"monitor": "loss"}) == ("loss", [])
    lipinc = zoo.build("lipinc_v2", 16)
    assert monitor_spec(lipinc, rows, {"monitor": "auc"}) == ("auc", ["visual"])
    unsup = zoo.build("avh_align", 16)  # chỉ học real: validation không có fake
    assert monitor_spec(unsup, rows, {"monitor": "auc"}) == ("loss", [])
    one_class = [r for r in rows if not r["label"]]
    assert monitor_spec(m3, one_class, {"monitor": "auc"})[0] == "loss"
    metrics = {"val_auc/audio": 0.9, "val_auc/mismatch": 0.8, "val_auc/visual": 0.7}
    assert monitor_value("auc", ["audio", "mismatch", "visual"], 5.0, metrics) == pytest.approx(0.8)
    assert monitor_value("auc", ["ai"], 5.0, metrics) is None
    assert monitor_value("loss", [], 5.0, metrics) == 5.0
    assert better(0.81, 0.80, "auc") and not better(0.79, 0.80, "auc")
    assert better(0.4, 0.5, "loss") and not better(0.6, 0.5, "loss")
    assert better(0.1, None, "auc") and not better(None, 0.5, "auc")


def test_scheduler_only_for_own_models_baselines_keep_original_recipe():
    import settings
    from vn_av_df.training.train import recipe

    cfg = settings.config()
    for name in ("trihead_v4", "trihead_v2", "avh_align_plus", "xlsr_frozen"):
        options = recipe(cfg, zoo.build(name, 16, 8, 0.0, assets={"n_speakers": 0}))
        assert options["scheduler"] == "plateau" and options["plateau_patience"] == 2
        assert options["patience"] == 5 and options["monitor"] == "auc"
    for name in ("lipinc_v2", "ssl_aasist", "avh_align_sup"):
        options = recipe(cfg, zoo.build(name, 16))
        assert options["scheduler"] is None and options["patience"] == 10
    options = recipe(cfg, zoo.build("avh_align", 16))  # gốc có plateau 5 theo val loss
    assert options["scheduler"] == "plateau" and options["plateau_patience"] == 5


def test_train_groups_cover_test_roster_once():
    import settings

    seen = [m for g in settings.TRAIN_GROUPS.values() for lane in g["lanes"] for m in lane]
    # mỗi model đúng một bước, mỗi bước một GPU (một lane)
    assert sorted(seen) == sorted(settings.TEST_MODELS)
    assert all(len(g["lanes"]) == 1 for g in settings.TRAIN_GROUPS.values())
    assert all(m in zoo.SPECS for m in seen)
    assert {g["tier"] for g in settings.TRAIN_GROUPS.values()} == {"core", "optional"}
    first = next(iter(settings.TRAIN_GROUPS.values()))
    assert first["benchmark"] and first["lanes"] == [["trihead_v5"]]
    steps = [s for lanes in settings.TRAIN_SESSIONS.values() for lane in lanes for s in lane]
    assert set(steps) == set(settings.TRAIN_GROUPS)  # 11/11: bỏ M1, M3 v004 khỏi đội hình
    assert all(len(lanes) >= 2 for lanes in settings.TRAIN_SESSIONS.values())
    # v005: P1 = M3-R (GPU0 riêng) + baseline lõi chia GPU1; P2 chạy tiếp M3-R, M2 + M3-R-nopair, BL-M
    assert settings.TRAIN_SESSIONS["P1"][0] == ["M3-R"]
    core = {k for k, g in settings.TRAIN_GROUPS.items() if g["tier"] == "core"}
    assert core <= {s for p in ("P1", "P2") for lane in settings.TRAIN_SESSIONS[p] for s in lane}
    assert {n for n, v in settings.VARIANTS.items() if v.get("train")} == {"c480", "c360"}
    assert settings.config()["training"]["variant_p"] == 0.5
    # mọi cổng R đều có đủ hai model trong đội hình (hoặc là ghép muộn)
    from vn_av_df.training.report import GATES

    for first, second, *_ in [g for g in GATES if g[3].startswith("R")]:
        assert {first, second} <= set(settings.TEST_MODELS) | {"bl_ens_av"}


def test_settings_label_rule_has_one_threshold():
    import settings

    spec = settings.config()["mismatch_labels"]
    assert spec["mode"] == "measured" and spec["meters"] == ["syncnet"]
    assert spec["negative_quantile"] == spec["positive_quantile"] == 0.95
    assert spec["sharpness_quantile"] == 0.0

    info = {"meters": ["syncnet"], "thr": {"syncnet": {"negative": 10.0, "positive": 10.0}}}

    def state(value):
        curve = [0.0] * 31
        curve[SL.MAX_LAG] = value
        return SL.measured_state({"bad": {"syncnet": curve}, "measurable": True}, 0, info)

    assert state(10.5) == 1 and state(10.0) == 0 and state(float("nan")) is None  # chỉ thiếu số đo mới mask


def test_architecture_record_describes_model():
    from vn_av_df.training.train import architecture

    model = zoo.build("trihead_v4", 16, 8, 0.0, assets={"n_speakers": 0, "ssl_groups": 2})
    arch = architecture(model, "trihead_v4", {"lr": 3e-4}, ("auc", ["visual"]))
    names = {m["module"] for m in arch["modules"]}
    assert {"sync", "prosody", "trace", "audio"} <= names
    assert arch["parameters"] == sum(m["parameters"] for m in arch["modules"])
    assert arch["recipe"] == "settings.training" and "tam giác" in arch["doc"]
    json.dumps(arch)  # ghi được vào W&B config / architecture.json
    base = architecture(zoo.build("lipinc_v2", 16), "lipinc_v2", {}, ("auc", ["visual"]))
    assert base["recipe"] == "code gốc (train_options)"


def test_setup_is_attached_to_the_session_run(tmp_path, monkeypatch):
    """Log setup ghi vào Output và đính kèm vào run của phiên, không có run W&B riêng."""
    import sys

    from fake_wandb import FakeWandb

    from vn_av_df.setup_log import log_setup
    from vn_av_df.training.train import merge_runs

    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "pip-install.log").write_text("Successfully installed torch\n", encoding="utf-8")
    worker = tmp_path / "workers" / ".kaggle-tools" / "avhubert"
    worker.mkdir(parents=True)
    (worker / "setup.log").write_text("x" * 3_000_000, encoding="utf-8")
    (worker / "environment.json").write_text(
        json.dumps({"status": "ready", "python": str(tmp_path / "none"), "signature": {}}),
        encoding="utf-8",
    )
    runs = tmp_path / "runs" / "train_all_v004"
    cfg = {
        "root": str(tmp_path),
        "runs": str(runs),
        "wandb": {"project": "p"},
        "train_group": "P1",
        "training": {"lr": 3e-4, "wandb_api_key": "secret"},
    }
    cfg["setup"] = log_setup(
        cfg,
        "train-P1-x",
        runs / "setup",
        logs=[logs / "pip-install.log"],
        env_root=tmp_path / "workers",
        info={"lanes": [["trihead_v4"]]},
        timings={"pip-install": 12.0},
    )
    assert fake.runs == []  # không mở run riêng
    folder = runs / "setup" / "train-P1-x"
    record = json.loads((folder / "setup.json").read_text(encoding="utf-8"))
    assert record["workers"]["avhubert"]["receipt"]["status"] == "ready"
    assert record["training"]["wandb_api_key"] == "***"  # không ghi giá trị secret
    assert (folder / "worker-avhubert-setup.log").stat().st_size == 2_000_000  # giữ phần cuối
    assert (folder / "pip-install.log").is_file() and (folder / "pip-freeze.txt").is_file()

    plain = tracking.open_run(cfg, "evaluate", "e")  # run phụ: không đính kèm
    assert "setup" not in plain.run.config and not plain.run.artifacts
    run = tracking.open_run(cfg, "train", "m", config={"session": "cũ"}, with_setup=True)
    assert run.run.config["setup"]["session"] == "train-P1-x"
    assert run.run.config["session"] == "train-P1-x"  # đổi được giá trị khi train tiếp phiên khác
    assert {"setup/kernel_packages", "setup/logs"} <= run.run.keys()
    assert run.run.step == 0  # không tăng step: epoch 1 log theo step=1 không bị W&B bỏ
    run.log({"epoch": 1}, step=1)
    assert run.run.artifacts[0].type == "setup-log"
    assert run.run.artifacts[0].name == "setup-train_all_v004-train-P1-x"
    # merge_runs giữ log setup của mọi phiên, không coi là thư mục model
    target = tmp_path / "merged"
    merge_runs(target, [runs])
    assert (target / "setup" / "train-P1-x" / "setup.json").is_file()


def test_mismatch_intervals_follow_labels():
    from vn_av_df.training.evaluate import intervals, localization

    sham = {"cell": "sham", "kind": "partial", "av_mismatch_intervals": [[1.0, 2.0]]}
    assert intervals(sham, "mismatch") == [(1.0, 2.0)]
    real = {"cell": "real", "kind": "full", "av_mismatch_intervals": []}
    assert intervals(real, "mismatch") == []

    def lipsync(state):
        info = {"mode": "measured", "gen_intervals": [[0.4, 1.2]]}
        return {
            "cell": "donor",
            "kind": "partial",
            "av_mismatch_intervals": [],
            "_mismatch": info,
        }, state

    import vn_av_df.training.sync_labels as SLmod

    original = SLmod.generated_state
    try:
        for state, expected in ((1, [(0.4, 1.2)]), (0, []), (None, None)):
            row, _ = lipsync(state)
            SLmod.generated_state = lambda info, k, s=state: s
            assert intervals(row, "mismatch") == expected
    finally:
        SLmod.generated_state = original

    class OneHead:
        heads = {"mismatch": "mismatch"}

    n = 100
    score = np.zeros(n)
    score[25:50] = 1.0  # đúng đoạn 1,0-2,0 s
    predictions = {"s": {"frames": {"mismatch": score}}}
    row = {**sham, "sample_id": "s"}
    out = localization(OneHead(), [row], predictions, "mismatch")
    assert out["clips"] == 1 and out["auc_inside_vs_outside"] == pytest.approx(1.0)
    assert out["by_cell"]["sham"]["clips"] == 1


# ---------------------------------------------------------------- kiểm ngữ điệu–đầu trước train
def test_prosody_audit_samples_per_group_and_reads_cache(tmp_path, capsys):
    from vn_av_df.common.runtime import save_npz
    from vn_av_df.training.prosody import audit, audit_sample

    rows = [
        {"sample_id": f"{cell}{i}", "split": "train", "cell": cell, "kind": "full"}
        for cell in ("real", "voice")
        for i in range(12)
    ]
    picked = audit_sample(rows, 5)
    assert len(picked) == 10 and {r["sample_id"] for r in picked} == {
        r["sample_id"] for r in audit_sample(list(reversed(rows)), 5)
    }  # không phụ thuộc thứ tự manifest
    assert len(audit_sample(rows, None)) == 24

    rng = np.random.default_rng(0)
    frames = 80
    for row in rows[:-1]:  # mẫu cuối thiếu cache: bỏ qua, có báo số thiếu
        sid = row["sample_id"]
        (tmp_path / "wave").mkdir(exist_ok=True)
        (tmp_path / "avhubert").mkdir(exist_ok=True)
        (tmp_path / "mouth").mkdir(exist_ok=True)
        wave = (rng.normal(size=(frames, 640)) * 3000).astype(np.int16)
        save_npz(tmp_path / "wave" / f"{sid}.npz", wave=wave)
        save_npz(
            tmp_path / "avhubert" / f"{sid}.npz",
            landmarks=rng.normal(size=(frames, 68, 2)).astype(np.float32),
            visual_valid=np.ones(frames, bool),
        )
        pose = rng.normal(size=(frames, 3)).astype(np.float32)
        save_npz(tmp_path / "mouth" / f"{sid}.npz", pose=pose)
    out = audit({"cache": [str(tmp_path)]}, rows, per_group=None, workers=2, every=10)
    assert out["train/voice/full"]["n"] == 11 and 0 <= out["train/voice/full"]["auc_vs_real"] <= 1
    printed = capsys.readouterr().out
    assert "24 / 24 mẫu" in printed and "10 / 24 mẫu" in printed and "thiếu cache: 1 mẫu" in printed


# ---------------------------------------------------------------- v005: M3-R + biến thể nén
def test_trihead_v5_quality_sham_and_consistency_losses():
    """M3-R: luồng visual chọn được, không đối kháng người nói, nhánh chất lượng (FiLM khởi tạo
    đồng nhất), loss sham / chất lượng / nhất quán khi batch có cặp sạch + nén."""
    assets = {"load_pretrained": False, "n_speakers": 2, "ssl_groups": 5}
    model = zoo.TriHeadV5(16, 8, 0.1, visual=("trace",), quality=True, sham_weight=1.0,
                          consistency=1.0, ssl_drop=0.1, assets=assets)
    assert model.visual_streams == ["trace"] and model.n_speakers == 0
    assert all(not f.weight.detach().any() for f in model.quality_film.values())
    batch = small_batch(B=4, groups=5)
    batch["cell_index"] = torch.tensor([1, 0, 1, 0])  # 2 cặp: [sạch sham, sạch real, nén sham, nén real]
    batch["degraded"] = torch.tensor([0.0, 0.0, 1.0, 1.0])
    batch["quality"] = torch.zeros(4, 60, 1)
    batch["pair"] = 2
    for key in ("speaker", "lag_shift", "offset_valid"):
        batch[key] = torch.cat([batch[key], batch[key]])
    model.train()
    out = model(batch)
    loss = model.loss(batch, out)
    assert {"sham_visual", "degrade", "consistency", "offset", "cell"} <= set(model.loss_parts)
    assert not any(k.startswith("speaker") for k in model.loss_parts)
    loss.backward()
    assert model.degrade_head[0].weight.grad is not None
    model.eval()
    del batch["pair"]
    assert "_pooled" not in model(batch)  # chỉ cần khi có cặp


def _variant_cache(root, sid, side, frames=50):
    for name, arrays in (
        ("avhubert", {"audio": np.zeros((frames, 16), np.float32), "visual": np.zeros((frames, 16), np.float32),
                      "audio_valid": np.ones(frames, bool), "visual_valid": np.ones(frames, bool)}),
        ("xlsr", {"ssl": np.zeros((frames, 5, 1024), np.float16)}),
        ("mouth", {"mouth": np.full((frames, 96, 96), side, np.uint8), "lips": np.zeros((frames, 20, 2), np.float16),
                   "pose": np.zeros((frames, 3), np.float16), "valid": np.ones(frames, bool),
                   "native_side": np.float32(side)}),
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
        np.savez(root / name / f"{sid}.npz", **arrays)


def test_bench_data_variants_pair_and_quality(tmp_path):
    """Biến thể nén: train chọn bản nén theo xác suất / theo cặp cùng ngẫu nhiên; đánh giá ép một
    biến thể; mẫu thiếu bản nén giữ bản gốc; khoá quality = log(native_side / 100)."""
    from vn_av_df.training.data import BenchData, collate

    clean, c480 = tmp_path / "clean", tmp_path / "c480"
    rows = []
    for sid in ("a_real", "b_real"):
        _variant_cache(clean, sid, 100)
        rows.append({"sample_id": sid, "split": "train", "cell": "real", "label": 0, "duration_s": 2.0,
                     "source_clip_id": sid, "speaker_id": "s"})
    _variant_cache(c480, "a_real", 66)  # b_real chưa có bản nén
    cfg = {"cache": [str(clean)], "training": {"audio_lag_aug": [-2, 5]}}
    needs = {"av", "ssl", "mouth"}
    data = BenchData(cfg, rows, needs, augment=True, lag_augment=True, crop_frames=30,
                     variants={"c480": [str(c480)]}, pair=True)
    assert data.variant_ids["c480"] == {"a_real"}
    item = data[0]
    pair = item["_pair"]
    assert item["degraded"] == 0.0 and pair["degraded"] == 1.0
    assert int(item["mouth"].mean()) != int(pair["mouth"].mean())  # đặc trưng từ bản nén
    assert item["lag_shift"] == pair["lag_shift"] and len(item["valid"]) == len(pair["valid"])
    assert np.allclose(pair["quality"], np.log(0.66)) and np.allclose(item["quality"], 0.0)
    batch = collate([data[0], data[0]])
    assert batch["pair"] == 2 and batch["mouth"].shape[0] == 4
    assert batch["degraded"].tolist() == [0.0, 0.0, 1.0, 1.0] and batch["quality"].shape[-1] == 1
    assert "_pair" not in data[1]  # thiếu bản nén → không có cặp
    always = BenchData(cfg, rows, needs, augment=True, variants={"c480": [str(c480)]}, variant_p=1.0)
    assert always[0]["degraded"] == 1.0 and always[1]["degraded"] == 0.0
    # Baseline không augment (BL-M/sup, BL-A2) vẫn phải nhận bản nén khi train.
    plain = BenchData(cfg, rows, needs, augment=False, variants={"c480": [str(c480)]}, variant_p=1.0)
    assert plain[0]["degraded"] == 1.0
    fixed = BenchData(cfg, rows[:1], needs, variants={"c480": [str(c480)]}, variant="c480")
    assert fixed[0]["degraded"] == 1.0
    with pytest.raises(ValueError, match="biến thể"):
        BenchData(cfg, rows, needs, variants={"c480": [str(c480)]}, variant="c360")


def test_report_variant_gate_reads_variant_predictions(tmp_path):
    """Cổng R trên bản nén đọc predictions-test-<biến thể>.npz; thiếu file thì bỏ cổng (không lỗi)."""
    from vn_av_df.training.evaluate import save_predictions
    from vn_av_df.training.report import GATES, load_scores

    folder = tmp_path / "trihead_v5_seed42"
    folder.mkdir()
    save_predictions(folder / "predictions-test-c480.npz", {"ai": {"a": 0.9, "b": 0.1}})
    assert load_scores(tmp_path, folder.name, "ai", "c480") == {"a": 0.9, "b": 0.1}
    assert load_scores(tmp_path, folder.name, "ai") == {}
    assert any(len(g) == 6 and g[5] == "c480" for g in GATES)
