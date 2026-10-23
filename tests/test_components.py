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
        'video_path="X"\naudio_path="Y"\nout_path="Z"\nmodel_path="M"\n'
        'DATA_FLAGS="--nframes 5 --sampling_batch_size=32 "\npython generate.py $GEN_FLAGS\n',
        encoding="utf-8",
    )
    envs = ("gen_diff2lip", "gen_iplap", "gen_vrt", "gen_latentsync")
    cfg = {"envs": dict.fromkeys(envs, "/env/python"), "generation": {}}
    args, out = lipsync.command(
        cfg, "diff2lip", repo, tmp_path / "face.mp4", tmp_path / "a.wav", tmp_path
    )
    script = (tmp_path / "run_diff2lip.sh").read_text(encoding="utf-8")
    assert (
        args[0] == "bash"
        and f'video_path="{tmp_path / "face.mp4"}"' in script
        and "model.pt" in script
        and f"--sample_path={tmp_path / 'd2l_gen'}" in script
        and "--sampling_batch_size=32" in script
    )
    lipsync.command(cfg, "diff2lip", repo, "f", "a", tmp_path, batch=8)
    assert "--sampling_batch_size=8 " in (tmp_path / "run_diff2lip.sh").read_text(encoding="utf-8")
    # Hai GPU chạy song song: thư mục tạm của upstream nằm riêng trong thư mục job.
    args, _ = lipsync.command(cfg, "videoretalking", repo, "f", "a", tmp_path)
    assert args[-3:] == ["--re_preprocess", "--tmp_dir", tmp_path.name]
    assert args[1].name == "vrt_worker.py"
    args, _ = lipsync.command(cfg, "latentsync_1_5", repo, "f", "a", tmp_path)
    assert args[-2:] == ["--temp_dir", tmp_path / "temp"]
    args, out = lipsync.command(
        cfg, "iplap", repo, tmp_path / "face.mp4", tmp_path / "a.wav", tmp_path
    )
    assert out.name == "result.avi" and args[1].name == "iplap_worker.py"
    with pytest.raises(FileNotFoundError, match="envs"):
        lipsync.command({"envs": {}, "generation": {}}, "videoretalking", repo, "f", "a", tmp_path)


def test_fit_donor_matches_parent_edges():
    """Donor nói 0,7 s / nghỉ 0,3 s: chọn đoạn nói vừa nhất, lặng đầu / cuối theo clip gốc."""
    rate = A.RATE
    t = np.arange(5 * rate) / rate
    noise = 1e-4 * np.random.default_rng(0).standard_normal(len(t))
    donor = (0.3 * np.sin(2 * np.pi * 200 * t) * (np.mod(t, 1.0) < 0.7) + noise).astype(np.float32)
    samples, lead, tail = 76 * 1920, int(0.1 * rate), int(0.3 * rate)
    out, window = A.fit_donor(donor, samples, lead, tail)
    assert len(out) == samples and window[1] - window[0] == pytest.approx(2.7)
    # Đoạn 2,7 s dài hơn chỗ trống 2,64 s: lặng cuối ngắn lại còn 0,24 s, lặng đầu giữ 0,1 s.
    got = np.array(A.edge_silence(out)) / rate
    assert got == pytest.approx([0.1, 0.24], abs=0.021)
    assert A.fit_error(window, samples, lead, tail) == pytest.approx(0.06, abs=1e-6)
    # Đoạn nói ngắn hơn chỗ trống: phần dư kéo dài chỗ nghỉ bên trong, lặng hai đầu vẫn đúng.
    out, window = A.fit_donor(donor, 2 * samples, lead, tail)
    assert len(out) == 2 * samples
    assert np.array(A.edge_silence(out)) / rate == pytest.approx([0.1, 0.3], abs=0.021)
    # Nói liền không chỗ nghỉ, dài hơn clip gốc: không đoạn nào vừa.
    steady = (0.3 * np.sin(2 * np.pi * 200 * t) + noise).astype(np.float32)
    assert A.fit_donor(steady, samples, lead, tail) == (None, None)


def test_fallback_paths_keep_old_records_valid():
    old = {"generator": "iplap", "version": "v", "checkpoint_sha256": "c"}
    new = {**old, "adapter_sha256": "new", "worker_sha256": "w"}
    assert lipsync.still_valid("iplap", {**old, "adapter_sha256": lipsync.OLD_ADAPTER}, new)
    assert not lipsync.still_valid("iplap", {**old, "adapter_sha256": "other"}, new)
    assert not lipsync.still_valid(
        "iplap", {**old, "version": "w", "adapter_sha256": lipsync.OLD_ADAPTER}, new
    )


def test_diff2lip_retries_smaller_batch_on_oom(tmp_path, monkeypatch):
    import re
    from pathlib import Path

    repo = tmp_path / "external/diff2lip"
    (repo / "scripts").mkdir(parents=True)
    (repo / "checkpoints").mkdir()
    (repo / "checkpoints/model.pt").write_bytes(b"x")
    (repo / "scripts/inference_single_video.sh").write_text(
        'out_path="Z"\nDATA_FLAGS="--sampling_batch_size=32 "\npython generate.py $GEN_FLAGS\n',
        encoding="utf-8",
    )
    batches = []

    def fake_run(args, cwd=None, timeout=None, log=None, env=None):
        script = Path(args[1]).read_text(encoding="utf-8")
        batches.append(int(re.search(r"--sampling_batch_size=(\d+)", script)[1]))
        if batches[-1] == 32:
            log.write_text("RuntimeError: CUDA out of memory. Tried to allocate 5.91 GiB")
            raise RuntimeError("Command failed (1)")
        Path(re.search(r'out_path="(.*)"', script)[1]).write_bytes(b"video")

    def fake_inputs(video, audio, folder, frames, width, height):
        return folder / "face.mp4", folder / "audio.wav"

    monkeypatch.setattr(lipsync, "run", fake_run)
    monkeypatch.setattr(lipsync, "prepare_inputs", fake_inputs)
    monkeypatch.setattr(lipsync, "normalize_output", lambda *a: batches.append("done"))
    cfg = {"root": str(tmp_path), "envs": {"gen_diff2lip": "/env/python"}, "generation": {}}
    lipsync.make_synthesize("diff2lip")(cfg, "v", "a", tmp_path / "out.mkv", 10, 64, 48)
    assert batches == [32, 8, "done"]


def test_adapter_output_decodes_with_exact_frames(tmp_path):
    from vn_av_df.common.runtime import run
    from vn_av_df.data.media import decode
    from vn_av_df.generation.generate import skip_reason

    produced, audio, out = tmp_path / "result.mp4", tmp_path / "audio.wav", tmp_path / "gen.mkv"
    lavfi = ("-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i")
    run([ffmpeg(), *lavfi, "testsrc2=s=96x64:r=25:d=0.32", "-pix_fmt", "yuv420p", produced])
    run([ffmpeg(), *lavfi, "sine=duration=0.4", "-ar", "16000", "-ac", "1", audio])
    lipsync.normalize_output(produced, audio, out, 10, 64, 48)
    assert decode(out)["frames"].shape[:3] == (10, 48, 64)
    error = RuntimeError(
        "Worker failed: iplap error: Traceback (most recent call last):\n"
        '  File "x.py", line 1\nValueError: bad face\n'
    )
    assert skip_reason(error) == (
        "generator_error",
        "iplap error: Traceback (most recent call last): … ValueError: bad face",
    )
    warning = "diff2lip error: " + "x" * 400 + "\nNameError: real cause\n"
    assert skip_reason(RuntimeError(f"Worker failed: {warning}"))[1].endswith(
        "NameError: real cause"
    )
    for tail in (
        "raise NotImplementedError  # not detect face\nNotImplementedError",
        "UnboundLocalError: local variable 'mask_sharp' referenced before assignment",
    ):
        assert skip_reason(RuntimeError(f"Worker failed: x error: {tail}"))[0] == "face"


def test_iplap_worker_keeps_result_half(tmp_path):
    """Upstream giả ghi [sketch | kết quả] rồi mux; worker giữ nửa phải, bỏ mux, dọn thư mục tạm."""
    import os
    import subprocess
    import sys
    import textwrap

    import cv2

    from vn_av_df.generation import iplap_worker

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "torch.py").write_text("def manual_seed(s):\n    pass\n", encoding="utf-8")
    (stubs / "torchvision").mkdir()
    (stubs / "torchvision/__init__.py").write_text("from . import models\n", encoding="utf-8")
    (stubs / "torchvision/models.py").write_text(
        "def vgg19(**kw):\n    assert 'pretrained' not in kw\n    return kw\n", encoding="utf-8"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inference_single.py").write_text(
        textwrap.dedent(
            """
            import argparse, os, subprocess
            import cv2, numpy as np, torchvision
            p = argparse.ArgumentParser()
            for k in ("--input", "--audio", "--output_dir"):
                p.add_argument(k)
            a = p.parse_args()
            assert os.path.isfile(a.input) and torchvision.models.vgg19(pretrained=True) == {}
            temp = "tempfile_of_" + a.output_dir.split("/")[-1]
            os.makedirs(temp)
            out = cv2.VideoWriter(temp + "/result.avi", cv2.VideoWriter_fourcc(*"DIVX"), 25, (64, 32))
            for i in range(5):
                frame = np.zeros((32, 64, 3), np.uint8)
                frame[:, 32:] = 200
                out.write(frame)
            out.release()
            assert subprocess.call("ffmpeg -i x " + temp + "/result.avi y.mp4", shell=True) == 0
            """
        ),
        encoding="utf-8",
    )
    video = tmp_path / "face.mp4"
    video.write_bytes(b"v")
    (tmp_path / "audio.wav").write_bytes(b"a")
    out = tmp_path / "result.avi"
    env = {**os.environ, "PYTHONPATH": str(stubs)}
    subprocess.run(
        [
            sys.executable,
            iplap_worker.__file__,
            "--video",
            video,
            "--audio",
            tmp_path / "audio.wav",
            "--output",
            out,
        ],
        cwd=repo,
        env=env,
        check=True,
    )
    stream = cv2.VideoCapture(str(out))
    frames = []
    while True:
        ok, frame = stream.read()
        if not ok:
            break
        frames.append(frame)
    assert len(frames) == 5 and frames[0].shape == (32, 32, 3) and frames[0].min() > 150
    assert sorted(p.name for p in repo.iterdir()) == ["inference_single.py"]


def test_iplap_worker_retries_on_crop_around_face(tmp_path):
    """FaceMesh giả trượt trên khung gốc: worker chạy lại trên khung quanh mặt rồi dán về."""
    import os
    import subprocess
    import sys
    import textwrap

    import cv2

    from vn_av_df.generation import iplap_worker

    stubs = tmp_path / "stubs"
    (stubs / "torchvision").mkdir(parents=True)
    (stubs / "torch.py").write_text("def manual_seed(s):\n    pass\n", encoding="utf-8")
    (stubs / "torchvision/__init__.py").write_text("from . import models\n", encoding="utf-8")
    (stubs / "torchvision/models.py").write_text(
        "def vgg19(**kw):\n    return kw\n", encoding="utf-8"
    )
    (stubs / "mediapipe.py").write_text(
        textwrap.dedent(
            """
            from types import SimpleNamespace as N
            box = N(xmin=0.4, ymin=0.3, width=0.2, height=0.3)
            class FaceDetection:
                def __init__(self, **kw):
                    assert kw["model_selection"] == 1
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    pass
                def process(self, image):
                    return N(detections=[N(location_data=N(relative_bounding_box=box))])
            solutions = N(face_detection=N(FaceDetection=FaceDetection))
            """
        ),
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inference_single.py").write_text(
        textwrap.dedent(
            """
            import argparse, os
            import cv2, numpy as np
            p = argparse.ArgumentParser()
            for k in ("--input", "--audio", "--output_dir"):
                p.add_argument(k)
            a = p.parse_args()
            stream, frames = cv2.VideoCapture(a.input), []
            while True:
                ok, f = stream.read()
                if not ok:
                    break
                frames.append(f)
            h, w = frames[0].shape[:2]
            if w > 40:
                raise NotImplementedError  # not detect face
            temp = "tempfile_of_" + a.output_dir.split("/")[-1]
            os.makedirs(temp)
            out = cv2.VideoWriter(temp + "/result.avi", cv2.VideoWriter_fourcc(*"DIVX"), 25, (2 * w, h))
            for f in frames:
                out.write(np.concatenate([f, np.full_like(f, 200)], 1))
            out.release()
            """
        ),
        encoding="utf-8",
    )
    video = tmp_path / "face.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 25, (64, 48))
    for _ in range(4):
        writer.write(np.full((48, 64, 3), 50, np.uint8))
    writer.release()
    (tmp_path / "audio.wav").write_bytes(b"a")
    out = tmp_path / "result.avi"
    worker = [sys.executable, iplap_worker.__file__, "--video", video, "--audio"]
    subprocess.run(
        [*worker, tmp_path / "audio.wav", "--output", out],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(stubs)},
        check=True,
    )
    frames = iplap_worker.read_frames(out)
    # Khung vuông: bao hộp mặt (12,8 x 14,4 px) + 1,2 cạnh mặt → 30 px (chẵn), tâm (32; 21,6).
    assert len(frames) == 4 and frames[0].shape == (48, 64, 3)
    assert frames[0][6:36, 17:47].min() > 150 and frames[0][0, 0, 0] == 50
    assert frames[0][:, :17].max() == 50 and frames[0][:, 47:].max() == 50
    assert sorted(p.name for p in repo.iterdir()) == ["inference_single.py"]


def test_vrt_worker_falls_back_to_s3fd_box(tmp_path):
    """dlib giả trượt: Croper lấy hộp S3FD rồi vẫn đặt landmark bằng predictor; cả hai trượt thì
    báo "Face not detected"."""
    import os
    import subprocess
    import sys
    import textwrap

    from vn_av_df.generation import vrt_worker

    stubs = tmp_path / "stubs"
    (stubs / "face_alignment/detection/sfd").mkdir(parents=True)
    for package in ("face_alignment", "face_alignment/detection"):
        (stubs / package / "__init__.py").write_text("", encoding="utf-8")
    (stubs / "face_alignment/detection/sfd/__init__.py").write_text(
        textwrap.dedent(
            """
            import os
            class FaceDetector:
                def __init__(self, device):
                    pass
                def detect_from_image(self, image):
                    assert image.shape == (8, 8, 3)
                    return [[1, 2, 5, 6, 0.9]] if os.environ.get("S3FD_FINDS") else []
            """
        ),
        encoding="utf-8",
    )
    (stubs / "torch.py").write_text(
        "class cuda:\n    is_available = staticmethod(lambda: False)\n", encoding="utf-8"
    )
    (stubs / "dlib.py").write_text(
        "class rectangle:\n    def __init__(self, *box):\n        self.box = box\n",
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    (repo / "utils").mkdir(parents=True)
    (repo / "utils/__init__.py").write_text("", encoding="utf-8")
    (repo / "utils/ffhq_preprocess.py").write_text(
        textwrap.dedent(
            """
            from types import SimpleNamespace as N
            class Croper:
                def predictor(self, image, rect):
                    return N(parts=lambda: [N(x=rect.box[0], y=rect.box[1])])
                def get_landmark(self, img_np):
                    return None  # dlib HOG trượt
                def crop(self, frames, xsize=512):
                    lm = self.get_landmark(frames[0])
                    if lm is None:
                        return None
                    return frames, lm.tolist(), None
            """
        ),
        encoding="utf-8",
    )
    (repo / "inference.py").write_text(
        textwrap.dedent(
            """
            import argparse
            import numpy as np
            from utils.ffhq_preprocess import Croper
            p = argparse.ArgumentParser()
            p.add_argument("--outfile")
            a = p.parse_args()
            frames, crop, quad = Croper().crop([np.zeros((8, 8, 3), np.uint8)])
            open(a.outfile, "w").write(str(crop))
            """
        ),
        encoding="utf-8",
    )
    out = tmp_path / "result.txt"
    command = [sys.executable, vrt_worker.__file__, "--outfile", out]
    env = {**os.environ, "PYTHONPATH": str(stubs), "S3FD_FINDS": "1"}
    subprocess.run(command, cwd=repo, env=env, check=True)
    assert out.read_text() == "[[1, 2]]"
    env.pop("S3FD_FINDS")
    failed = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True)
    assert failed.returncode and "Face not detected" in failed.stderr


def test_wandb_failure_does_not_stop_run(monkeypatch, capsys):
    """Key / quyền W&B sai: cảnh báo rồi chạy tiếp không log (file local vẫn là bằng chứng)."""
    import sys
    import types

    from vn_av_df import tracking

    fake = types.ModuleType("wandb")

    def init(**kwargs):
        raise RuntimeError("the provided API key cannot access this resource")

    fake.init = init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    run = tracking.open_run({"wandb": {"project": "p"}}, "prepare", "x")
    assert not run.active
    run.log({"a": 1})
    run.summary({"b": 2})
    run.finish()
    assert "CẢNH BÁO W&B" in capsys.readouterr().out


def test_environment_and_lipinc_fallback_count(tmp_path):
    from vn_av_df.common.runtime import environment, write_json
    from vn_av_df.training.prepare import cache_quality

    env = environment()
    assert env["python"] and "numpy" in env["packages"] and isinstance(env["gpus"], list)
    rows = [
        {"sample_id": "a", "cell": "real"},
        {"sample_id": "b", "cell": "source"},
        {"sample_id": "c", "cell": "source"},  # chưa trích: không tính
    ]
    write_json(tmp_path / "a.json", {"fallback": True})
    write_json(tmp_path / "b.json", {"fallback": False})
    count = cache_quality("lipinc", tmp_path, rows)["lipinc_fallback"]
    assert count["samples"] == 2 and count["fallback"] == 1 and count["ratio"] == 0.5
    assert count["ratio_by_cell"] == {"real": 1.0, "source": 0.0}
    assert cache_quality("avhubert", tmp_path, rows) == {}


def test_progress_prints_first_periodic_last_and_heartbeat(capsys):
    """Tiến độ: mục đầu (gồm khởi động), mỗi every_n, mục cuối; luồng nền in "vẫn chạy"."""
    import time

    from vn_av_df.progress import Progress, duration, system_status

    with Progress("x", 5, every_n=2, every_s=3600) as p:
        for _ in range(5):
            p.step()
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("x:")]
    assert [x.split()[1] for x in lines] == ["1/5", "2/5", "4/5", "5/5"]
    assert "gồm khởi động" in lines[0]
    with Progress("y", 3, every_s=0.05):
        time.sleep(0.3)
    assert "y: vẫn chạy, 0/3" in capsys.readouterr().out
    assert isinstance(system_status(), str) and duration(4000) == "1 giờ 06 phút"


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
