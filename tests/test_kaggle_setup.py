"""Kiểm bootstrap không tải package thật: isolation, resume, failure và archive safety."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "bootstrap_worker", Path(__file__).resolve().parents[1] / "environments/bootstrap_worker.py"
)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


@pytest.fixture
def installer(tmp_path, monkeypatch):
    """Giả lập process con, giữ filesystem/receipt thật để kiểm khi bị ngắt."""
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(bootstrap, "micromamba", lambda path: path / "micromamba")
    monkeypatch.setattr(bootstrap.subprocess, "check_output", lambda *a, **k: "numpy==1.23.5\n")
    for worker, source in (("avhubert", "av_hubert"), ("musetalk", "MuseTalk")):
        for relative in (
            f"environments/{worker}.yml",
            f"external/{source}/vn_av_revision.json",
            f"external/{source}/requirements.txt",
            f"external/{source}/fairseq/setup.py",
            f"external/{source}/avhubert/hubert.py",
            f"external/{source}/musetalk/utils/utils.py",
        ):
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture", encoding="utf-8")
    commands = []

    def execute(command, cwd, log, env=None):
        command = [str(x) for x in command]
        commands.append((command, env))
        if "-p" in command:
            python = Path(command[command.index("-p") + 1]) / "bin/python"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("fixture")
        if "-c" in command:
            worker = Path(command[0]).parents[1].name.removeprefix(".venv-")
            path = tmp_path / ".kaggle-tools" / worker / "probe.json"
            path.write_text(json.dumps({"cuda_available": True}))

    monkeypatch.setattr(bootstrap, "execute", execute)
    return tmp_path, commands, execute


@pytest.mark.parametrize("worker", ["avhubert", "musetalk"])
def test_worker_isolated_and_ready_resume_rechecks_without_reinstall(
    installer, monkeypatch, worker
):
    root, commands, _ = installer
    monkeypatch.setenv("PYTHONPATH", "notebook-only-packages")
    monkeypatch.setenv("PIP_TARGET", "do-not-write-here")
    python = bootstrap.setup_worker(root, worker)
    assert python == root / f".venv-{worker}/bin/python"
    assert all("PYTHONPATH" not in env and "PIP_TARGET" not in env for _, env in commands)
    for command, _ in commands:
        if "pip" in command:
            assert command[0] == str(python)
    installed = "\n".join(" ".join(c) for c, _ in commands)
    assert ("mmpose==1.1.0" in installed) == (worker == "musetalk")
    assert ("fairseq" in installed) == (worker == "avhubert")
    receipt = root / ".kaggle-tools" / worker / "environment.json"
    assert json.loads(receipt.read_text())["status"] == "ready"
    commands.clear()
    bootstrap.setup_worker(root, worker)
    assert len(commands) == 2  # pip check và import/CUDA probe vẫn phải chạy lại.
    assert all("install" not in command for command, _ in commands)
    (root / "environments" / f"{worker}.yml").write_text("changed")
    with pytest.raises(ValueError, match="specification/source changed"):
        bootstrap.setup_worker(root, worker)


def test_failed_install_resumes_but_failed_probe_is_not_ready(installer, monkeypatch):
    root, commands, real_execute = installer

    def fail(command, *args, **kwargs):
        if "--index-url" in command:
            raise RuntimeError("interrupted package download")
        return real_execute(command, *args, **kwargs)

    monkeypatch.setattr(bootstrap, "execute", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        bootstrap.setup_worker(root, "avhubert")
    receipt = root / ".kaggle-tools/avhubert/environment.json"
    assert json.loads(receipt.read_text())["status"] == "installing"
    monkeypatch.setattr(bootstrap, "execute", real_execute)
    commands.clear()
    bootstrap.setup_worker(root, "avhubert")
    assert "install" in commands[0][0]  # Môi trường đã tạo, tiếp tục dependency còn dở.

    def fail_probe(command, *args, **kwargs):
        if "-c" in command:
            raise RuntimeError("CUDA unavailable")
        return real_execute(command, *args, **kwargs)

    monkeypatch.setattr(bootstrap, "execute", fail_probe)
    with pytest.raises(RuntimeError, match="CUDA"):
        bootstrap.setup_worker(root, "avhubert")
    assert json.loads(receipt.read_text())["status"] != "ready"


def test_missing_source_unmanaged_environment_and_non_linux_fail_early(installer, monkeypatch):
    root, commands, _ = installer
    source = root / "external/av_hubert/fairseq/setup.py"
    source.unlink()
    with pytest.raises(FileNotFoundError, match="asset setup"):
        bootstrap.setup_worker(root, "avhubert")
    (root / ".venv-musetalk").mkdir()
    with pytest.raises(ValueError, match="unmanaged"):
        bootstrap.setup_worker(root, "musetalk")
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Kaggle/Linux"):
        bootstrap.setup_worker(root, "musetalk")
    assert not commands


def test_archive_extraction_never_writes_other_members(tmp_path):
    archive, target = tmp_path / "package.tar.bz2", tmp_path / "micromamba"
    with tarfile.open(archive, "w:bz2") as package:
        for name in ("../escaped", "bin/micromamba"):
            member = tarfile.TarInfo(name)
            member.size = 7
            package.addfile(member, io.BytesIO(b"fixture"))
    bootstrap.extract_micromamba(archive, target)
    assert target.read_bytes() == b"fixture"
    assert not (tmp_path.parent / "escaped").exists()
    with tarfile.open(archive, "w:bz2") as package:
        member = tarfile.TarInfo("bin/micromamba")
        member.type, member.linkname = tarfile.SYMTYPE, "../elsewhere"
        package.addfile(member)
    with pytest.raises(ValueError, match="regular file"):
        bootstrap.extract_micromamba(archive, target)


@pytest.mark.parametrize("worker", ["avhubert", "musetalk"])
def test_generated_probe_is_valid_python(worker, tmp_path):
    compile(bootstrap.probe_code(worker, tmp_path, tmp_path / "probe.json", True), "probe", "exec")


@pytest.mark.parametrize("notebook,worker", [("generate", "musetalk"), ("train", "avhubert")])
@pytest.mark.parametrize("needed", [True, False])
def test_notebook_bootstrap_configures_only_its_worker(notebook, worker, tmp_path, needed):
    path = Path(__file__).resolve().parents[1] / "notebooks" / f"{notebook}.ipynb"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(path), "exec")
    setup = next(c for c in cells if c["id"] == notebook + "-worker-setup")
    commands = []
    cfg = {
        "musetalk": {"python": "unchanged"},
        "encoders": {"avhubert": {"python": "unchanged"}},
        "architectures": ["selected"],
        "methods": {"selected": {"encoder": "avhubert" if needed else "fate"}},
        "generation": {
            "generators_by_split": {"test": ["musetalk_1_5" if needed else "wav2lip_gan"]}
        },
    }
    state = dict(
        ROOT=tmp_path,
        cfg=cfg,
        settings=SimpleNamespace(),
        subprocess=SimpleNamespace(run=lambda args, **kwargs: commands.append((args, kwargs))),
        sys=SimpleNamespace(executable="notebook-python"),
    )
    exec(compile("".join(setup["source"]), "worker-cell", "exec"), state)
    if not needed:
        assert not commands
        assert cfg["musetalk"]["python"] == cfg["encoders"]["avhubert"]["python"] == "unchanged"
        return
    expected = str(tmp_path / f".venv-{worker}/bin/python")
    assert commands == [
        (
            [
                "notebook-python",
                str(tmp_path / "environments/bootstrap_worker.py"),
                worker,
                "--root",
                str(tmp_path),
            ],
            {"check": True},
        )
    ]
    assert cfg["musetalk"]["python"] == (expected if worker == "musetalk" else "unchanged")
    assert cfg["encoders"]["avhubert"]["python"] == (
        expected if worker == "avhubert" else "unchanged"
    )


def notebook_config_state(notebook, tmp_path, monkeypatch):
    """Thực thi cell cấu hình thật, thay các action download/train bằng recorder."""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("notebook_settings_fixture", root / "settings.py")
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    cells = json.loads((root / "notebooks" / f"{notebook}.ipynb").read_text(encoding="utf-8"))[
        "cells"
    ]
    sources = {c["id"]: "".join(c["source"]) for c in cells if c["cell_type"] == "code"}
    calls = []
    state = dict(
        ROOT=tmp_path,
        Path=Path,
        settings=settings,
        execute=lambda action, cfg: calls.append(action),
    )
    exec(sources[f"{notebook}-02"], state)
    return sources, state, calls


@pytest.mark.parametrize("selection", ["all", ["fate_gru"], ["fate_linear", "p2_sync_only"]])
def test_training_notebook_uses_edited_models_parts_and_hyperparameters(
    tmp_path, monkeypatch, selection
):
    from vn_av_df import dataset

    monkeypatch.setattr(dataset, "training_dataset", lambda *args, **kwargs: ([], {"samples": 0}))
    sources, state, calls = notebook_config_state("train", tmp_path, monkeypatch)
    exec(sources["train-parameters"], state)
    state.update(
        ARCHITECTURES=selection, SEEDS=[99], DATASET_PARTS=["chosen_part"], RUN_NAME="custom_run"
    )
    state["TRAINING"].update(epochs=3, lr=0.002, hidden=32)
    state["RECONSTRUCTION"]["epochs"] = 2
    state["SYNC"]["shift_frames"] = [2, 4]
    state["DINO_FEATURES"]["crop_size"] = 112
    exec(sources["train-config"], state)
    cfg = state["cfg"]
    assert cfg["architectures"] == (
        ["fate_gru", "avh_tcn", "p2_syncartifact"] if selection == "all" else selection
    )
    assert cfg["dataset_parts"] == ["chosen_part"] and cfg["seeds"] == [99]
    assert Path(cfg["runs"]).name == "custom_run"
    assert cfg["training"]["lr"] == 0.002 and cfg["training"]["epochs"] == 3
    assert cfg["reconstruction"]["epochs"] == 2
    assert cfg["sync"]["shift_frames"] == [2, 4]
    assert cfg["encoders"]["dinov2"]["crop_size"] == 112
    assert calls == ["encoder_setup"]
    assert state["RUN_TEST"] is False


@pytest.mark.parametrize("use_history", [False, True])
def test_generation_notebook_uses_edited_part_and_generator_settings(
    tmp_path, monkeypatch, use_history
):
    sources, state, calls = notebook_config_state("generate", tmp_path, monkeypatch)
    state.update(
        PART=2,
        GENERATION_NAME="vn-av-df-data-part2",
        CLIPS_PER_SPLIT=0,
        CLEAN_DATASET=tmp_path / "clean2",
        SPLIT_HISTORY=[tmp_path / "prior.json"] if use_history else [],
        CRF=20,
        MUSETALK_BATCH_SIZE=2,
        PARTIAL_SECONDS=[0.8],
    )
    state["settings"].ROOT = tmp_path
    exec(sources["generate-config"], state)
    cfg = state["cfg"]
    assert cfg["data_part"] == 2 and cfg["clean_dataset"] == str(tmp_path / "clean2")
    assert Path(cfg["generated_dataset"]).name == "vn-av-df-data-part2"
    assert cfg["generation"]["clips_per_split"] == 0 and cfg["generation"]["crf"] == 20
    assert cfg["generation"]["partial_seconds"] == [0.8]
    assert cfg["generation"]["fake_audio_modes_by_split"]["train"] == ["donor"]
    assert "audio_mode" not in cfg["generation"]
    assert cfg["musetalk"]["batch_size"] == 2
    assert cfg["split_history"] == ([str(tmp_path / "prior.json")] if use_history else [])
    assert cfg["split_ratios"] == {"train": 0.8, "validation": 0.1, "test": 0.1}
    assert calls == ["validate", "plan", "generator_setup"]
