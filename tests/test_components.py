"""Các phần nhỏ của v003: sửa cụm từ, ghép tiếng, encoder mới, lệnh generator."""

import numpy as np
import pytest
import torch

from vn_av_df.data.media import ffmpeg
from vn_av_df.generation import lipsync
from vn_av_df.generation.voice import audio as A
from vn_av_df.generation.voice.edit import choose_edit, edit_words


def words(text, step=0.4):
    return [
        {"word": w, "start": 0.6 + i * step, "end": 0.6 + i * step + 0.35}
        for i, w in enumerate(text.split())
    ]


def test_choose_edit_prefers_meaning_flip_then_numbers():
    edit = choose_edit(words("hôm nay giá xăng không tăng nhiều lắm"), 6.0, seed=1)
    assert edit is not None and 0.5 <= edit["span"][1] - edit["span"][0] <= 3.0
    assert edit["kind"].startswith("swap") and edit["new"] != edit["old"]
    import random

    new, kind = edit_words(["giảm", "hai", "nghìn"], random.Random(0))
    assert kind == "swap:giảm->tăng" and new.startswith("tăng")
    assert edit_words(["tốn", "3", "triệu"], random.Random(0))[1] == "number"
    assert choose_edit(words("a"), 6.0, seed=0) is None  # quá ngắn để có cụm 0,5 s


def test_splice_and_stretch_keep_length_and_level():
    rate = A.RATE
    base = 0.1 * np.sin(2 * np.pi * 200 * np.arange(rate * 3) / rate).astype(np.float32)
    insert = 0.8 * np.sin(2 * np.pi * 300 * np.arange(rate) / rate).astype(np.float32)
    stretched, ratio = A.time_stretch(insert, rate // 2, ffmpeg())
    assert len(stretched) == rate // 2 and ratio == pytest.approx(2.0)
    out = A.splice(base, stretched, rate, rate + rate // 2)
    assert len(out) == len(base)
    assert A.rms(out[rate + 2000 : rate + rate // 2 - 2000]) == pytest.approx(
        A.rms(base[rate : rate + rate // 2]), rel=0.15
    )
    assert np.allclose(out[: rate - 1000], base[: rate - 1000])
    fitted, _ = A.fit_whole(insert, rate * 2, ffmpeg())
    assert len(fitted) == rate * 2


def test_mouth_encoder_crops_native_resolution(tmp_path):
    from vn_av_df.common.runtime import run, save_npz, sha, write_json
    from vn_av_df.features.encoders import MouthEncoder, normalized_lips

    video = tmp_path / "v.mp4"
    run(
        [
            ffmpeg(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=1280x720:r=25:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            video,
        ]
    )
    n = 50
    boxes = np.tile([[320.0, 180.0, 50.0]], (n, 1)).astype(np.float32)
    box = tmp_path / "box.npz"
    save_npz(
        box,
        mouth_boxes=boxes,
        landmarks=np.random.default_rng(0).normal(100, 3, (n, 68, 2)),
        visual_valid=np.ones(n, bool),
    )
    write_json(box.with_suffix(".json"), {"feature_sha256": sha(box)})
    meta = MouthEncoder({"crop": 64}).extract(
        {"video": str(video), "box_cache": str(box)}, tmp_path / "m.npz"
    )
    with np.load(tmp_path / "m.npz") as z:
        assert z["mouth"].shape == (n, 64, 64) and z["lips"].shape == (n, 20, 2)
        assert float(z["native_side"]) == pytest.approx(100.0)  # 50 px ở 640 → 100 px ở 1280
        assert z["pose"].shape == (n, 3) and np.isfinite(z["pose"]).all()
    assert meta["box_sha256"] == sha(box)
    lips = normalized_lips(np.random.default_rng(1).normal(0, 5, (3, 68, 2)))
    assert np.allclose(lips.astype(np.float32).mean(1), 0, atol=1e-2)


def synthetic_video(path, seconds=2, size="64x64"):
    from vn_av_df.common.runtime import run

    run(
        [
            ffmpeg(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s={size}:r=25:d={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=duration={seconds}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            path,
        ]
    )
    return path


def face_points(opening):
    """68 landmark dlib giả, mặt thẳng; opening = khoảng cách môi trong 62-66 (px ở khung 640)."""
    p = np.zeros((68, 2), np.float32)
    p[0:17, 0] = np.linspace(260, 380, 17)
    p[0:17, 1] = 200 + 60 * np.sin(np.linspace(0, np.pi, 17))
    p[17:27] = np.c_[np.linspace(270, 370, 10), np.full(10, 150)]
    p[36:42] = np.c_[np.linspace(285, 305, 6), np.full(6, 170)]
    p[42:48] = np.c_[np.linspace(335, 355, 6), np.full(6, 170)]
    p[30] = (320, 200)
    p[48:60] = np.c_[
        300 + 40 * np.cos(np.linspace(0, 2 * np.pi, 12)),
        230 + 8 * np.sin(np.linspace(0, 2 * np.pi, 12)),
    ]
    p[60:68] = np.c_[
        305 + 30 * np.cos(np.linspace(0, 2 * np.pi, 8)),
        230 + 3 * np.sin(np.linspace(0, 2 * np.pi, 8)),
    ]
    p[48], p[54], p[60], p[64] = (285, 230), (355, 230), (292, 230), (348, 230)
    p[62], p[66] = (320, 230 - opening / 2), (320, 230 + opening / 2)
    p[8] = (320, 262)
    return p


def test_lipinc_selection_follows_original_rule():
    from vn_av_df.features.encoders import lipinc_select

    # Frame 0-1 miệng khép (< 1 %) cắt chuỗi; 2-6 là 5 frame mở liên tiếp; frame cùng tư thế cách ≥ 3.
    ratios = [(i, (0.005, 0.3, 0.2, 0.2)) for i in range(2)]
    ratios += [(i, (0.05, 0.30, 0.20, 0.20)) for i in range(2, 7)]
    ratios += [(i, (0.20, 0.50, 0.40, 0.40)) for i in range(7, 12)]
    ratios += [(i, (0.051, 0.301, 0.201, 0.201)) for i in range(12, 30)]
    local, found, fallback = lipinc_select(ratios)
    assert not fallback and local == [2, 3, 4, 5, 6] and len(found) == 3
    assert all(b - a >= 4 for a, b in zip(found, found[1:]))  # sau mỗi frame chọn bỏ qua 3 frame
    # Không có frame cùng tư thế: vẫn chọn theo luật dự phòng, đánh dấu fallback.
    lone = [(i, (0.05 + 0.03 * i, 0.3 + 0.05 * i, 0.2, 0.2)) for i in range(12)]
    local, found, fallback = lipinc_select(lone)
    assert fallback and len(local) == 5 and len(found) == 3


def test_lipinc_and_wave_encoders(tmp_path):
    from vn_av_df.common.runtime import save_npz, sha, write_json
    from vn_av_df.features.encoders import LipincEncoder, WaveEncoder

    video = synthetic_video(tmp_path / "v.mp4", size="1280x720")
    n = 50
    landmarks = np.stack([face_points(4 + 6 * (i % 7 > 1)) for i in range(n)])
    box = tmp_path / "box.npz"
    save_npz(box, landmarks=landmarks, visual_valid=np.ones(n, bool))
    write_json(box.with_suffix(".json"), {"feature_sha256": sha(box)})
    meta = LipincEncoder({}).extract(
        {"video": str(video), "box_cache": str(box)}, tmp_path / "l.npz"
    )
    with np.load(tmp_path / "l.npz") as z:
        assert z["frames"].shape == (8, 64, 144, 3) and z["frames"].dtype == np.uint8
        assert (z["frame_ids"] >= 0).sum() == meta["selected"] == 8
        assert z["frames"].reshape(8, -1).std(1).min() > 0  # mọi ô đều có ảnh
    WaveEncoder({}).extract({"video": str(video)}, tmp_path / "w.npz")
    with np.load(tmp_path / "w.npz") as z:
        assert z["wave"].shape[0] in (50, 51) and z["wave"].shape[1] == 640  # AAC đệm thêm mẫu
        assert z["wave"].dtype == np.int16
        assert np.abs(z["wave"]).max() > 1000  # sóng sin, không chuẩn hoá


def test_xlsr_encoder_pools_layers_to_25hz(tmp_path):
    from types import SimpleNamespace

    from vn_av_df.common.runtime import run
    from vn_av_df.features.encoders import XLSREncoder

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, output_hidden_states=True):
            frames = x.shape[1] // 320
            return SimpleNamespace(
                hidden_states=[torch.full((1, frames, 1024), float(i)) for i in range(25)]
            )

    video = tmp_path / "v.mp4"
    run(
        [
            ffmpeg(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x64:r=25:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            video,
        ]
    )
    encoder = XLSREncoder({"layer_groups": [[1, 4], [5, 8], [9, 12]]}, model=Fake())
    encoder.extract({"video": str(video)}, tmp_path / "x.npz")
    with np.load(tmp_path / "x.npz") as z:
        ssl = z["ssl"].astype(np.float32)
        assert ssl.shape[0] in (50, 51) and ssl.shape[1:] == (3, 1024)
        assert np.allclose(ssl[:, :, 0].mean(0), [2.5, 6.5, 10.5])  # TB các lớp trong từng nhóm


def test_npr_matches_reference_pixel_definition():
    from vn_av_df.training.models import trace_streams

    mouth = torch.randint(0, 255, (1, 3, 8, 8), dtype=torch.uint8)
    npr = trace_streams(mouth)[0, 1]
    x = mouth.float() / 255
    expected = (x - x[..., ::2, ::2].repeat_interleave(2, -2).repeat_interleave(2, -1)) * 4
    assert torch.allclose(npr, expected[0]) and torch.all(npr[..., ::2, ::2] == 0)


def test_gradient_reversal_flips_gradient():
    from vn_av_df.training.models import GradReverse

    x = torch.ones(3, requires_grad=True)
    GradReverse.apply(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.5))


def test_pseudo_fake_replaces_audio_segment_only():
    import random

    from vn_av_df.training.data import replace_audio

    n = 150
    item = {"a": np.zeros((n, 4), np.float32), "v": np.zeros((n, 4), np.float32)}
    other = {"a": np.ones((n, 4), np.float32), "v": np.ones((n, 4), np.float32)}
    targets = {"mismatch": np.zeros(n, np.float32)}
    start, end = replace_audio(item, other, targets, random.Random(3))
    assert 12 <= start and end <= n - 12 and 7 <= end - start <= 75
    assert item["a"][start:end].all() and not item["a"][:start].any() and not item["v"].any()
    assert targets["mismatch"][start:end].all() and targets["mismatch"].sum() == end - start


def test_generator_commands(tmp_path):
    repo = tmp_path / "diff2lip"
    (repo / "scripts").mkdir(parents=True)
    (repo / "checkpoints").mkdir()
    (repo / "checkpoints/model.pt").write_bytes(b"x")
    (repo / "scripts/inference_single_video.sh").write_text(
        'video_path="X"\naudio_path="Y"\nout_path="Z"\nmodel_path="M"\npython generate.py $video_path\n',
        encoding="utf-8",
    )
    cfg = {"envs": {"gen_diff2lip": "/env/python", "gen_iplap": "/env/py37"}, "generation": {}}
    args, out = lipsync.command(
        cfg, "diff2lip", repo, tmp_path / "face.mp4", tmp_path / "a.wav", tmp_path
    )
    script = (tmp_path / "run_diff2lip.sh").read_text(encoding="utf-8")
    assert (
        args[0] == "bash"
        and f'video_path="{tmp_path / "face.mp4"}"' in script
        and "model.pt" in script
    )
    args, out = lipsync.command(
        cfg, "iplap", repo, tmp_path / "face.mp4", tmp_path / "a.wav", tmp_path
    )
    assert out.name == "faceresult_N_25_Nl_15.mp4" and "--output_dir" in [str(a) for a in args]
    with pytest.raises(FileNotFoundError, match="envs"):
        lipsync.command({"envs": {}, "generation": {}}, "videoretalking", repo, "f", "a", tmp_path)


def test_env_recipes_cover_every_worker():
    import sys
    from pathlib import Path

    from vn_av_df.generation.voice.engines import ENGINES

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environments"))
    from worker_envs import RECIPES

    needed = {env for steps in ENGINES.values() for env, _, _ in steps} | set(lipsync.ENVS.values())
    assert needed <= set(RECIPES)


@pytest.mark.parametrize("lag", [2, -1])
def test_lag_audio_shifts_audio_features_and_masks_edges(lag):
    from vn_av_df.training.data import lag_audio

    t = np.arange(10, dtype=np.float32)[:, None]
    item = {"a": t.copy(), "ssl": t.copy() * 2, "v": t.copy()}
    out = lag_audio(item, lag)
    valid = out["lag_valid"]
    assert valid.sum() == 10 - abs(lag)
    # Ở frame còn hợp lệ: tiếng tại t là tiếng gốc của t - lag (tiếng trễ lag frame).
    assert np.all(out["a"][valid, 0] == np.arange(10)[valid] - lag)
    assert np.all(out["ssl"][valid, 0] == 2 * (np.arange(10)[valid] - lag))
    assert np.all(out["v"] == t)  # hình giữ nguyên


@pytest.mark.parametrize(
    "clips,generators,per_clip",
    [(600, 4, 2), (240, 6, 3), (12, 3, 3), (160, 4, 2)],  # train, test, cũ (đủ), validation
)
def test_generator_assignment_balanced_and_varied(clips, generators, per_clip):
    from collections import Counter

    from vn_av_df.generation.plan import assign_generators

    names = [f"g{j}" for j in range(generators)]
    plan = assign_generators(clips, names, per_clip, "7:train")
    assert len(plan) == clips and all(len(s) == per_clip for s in plan)
    runs = Counter(g for s in plan for g, _ in s)
    assert set(runs.values()) == {clips * per_clip // generators}  # mỗi generator chạy bằng nhau
    for s in plan:  # trong một clip: generator khác nhau, ô khác nhau
        assert len({g for g, _ in s}) == per_clip and len({c for _, c in s}) == min(per_clip, 3)
    cells = Counter((g, c) for s in plan for g, c in s)
    for g in names:
        counts = [cells[(g, c)] for c in range(3)]
        assert max(counts) - min(counts) <= 1  # mỗi generator chia đều 3 ô
    if per_clip < generators:
        pairs = Counter(tuple(sorted(g for g, _ in s)) for s in plan)
        assert len(pairs) > generators // per_clip  # bộ generator chung clip không cố định


def test_temporal_metrics_perfect_and_empty():
    from vn_av_df.training.evaluate import proposals, temporal_metrics

    class Model:
        heads = {"score": "ai"}

    rows, predictions = [], {}
    for i, span in enumerate([[1.0, 2.0], [2.4, 3.0], None]):
        scores = np.zeros(125)
        if span:
            scores[int(span[0] * 25) : int(span[1] * 25)] = 1.0
        rows.append({"sample_id": f"s{i}", "fake_intervals": [span] if span else []})
        predictions[f"s{i}"] = {"frames": {"score": scores}, "video": {}}
    out = temporal_metrics(Model(), rows, predictions, "ai")
    assert out["segments"] == 2 and out["ap@0.5"] == pytest.approx(1.0)
    assert out["ap@0.95"] == pytest.approx(1.0) and out["ar@1"] == pytest.approx(1.0)
    assert proposals(np.zeros(50)) == [] or all(b > a for _, a, b in proposals(np.zeros(50)))
    flat = {"s0": {"frames": {"score": np.random.default_rng(0).random(125)}, "video": {}}}
    noisy = temporal_metrics(Model(), rows[:1], flat, "ai")
    assert 0 <= noisy["ap@0.5"] < 1


def test_speaker_balance_weights_and_auc():
    from vn_av_df.training.evaluate import speaker_balanced_auc
    from vn_av_df.training.train import speaker_weights

    def rows(speaker, clips):
        return [
            {
                "split": "train",
                "speaker_id": speaker,
                "source_clip_id": f"{speaker}{i}",
                "sample_id": f"{speaker}{i}{k}",
            }
            for i in range(clips)
            for k in ("real", "fake")
        ]

    same = rows("a", 3) + rows("b", 3)
    assert speaker_weights(same) is None  # cùng số clip: giữ shuffle thường
    mixed = rows("a", 2) + rows("b", 4)
    weights = speaker_weights(mixed)
    # Tổng trọng số mỗi người bằng nhau dù số clip khác nhau.
    assert sum(w for w, r in zip(weights, mixed) if r["speaker_id"] == "a") == pytest.approx(
        sum(w for w, r in zip(weights, mixed) if r["speaker_id"] == "b")
    )
    labels = [int(r["sample_id"].endswith("fake")) for r in mixed]
    # Người a tách hoàn hảo, người b ngẫu nhiên đảo: AUC cân người khác AUC gộp thường.
    scores = [
        (0.9 if y else 0.1) if r["speaker_id"] == "a" else (0.4 if y else 0.6)
        for y, r in zip(labels, mixed)
    ]
    balanced = speaker_balanced_auc(mixed, labels, scores)
    assert 0.0 <= balanced <= 1.0 and speaker_balanced_auc(mixed, [0] * len(mixed), scores) is None
