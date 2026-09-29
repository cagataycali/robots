"""Every detached lerobot launcher runs THIS interpreter, never a bare ``python`` from PATH.

A host with only ``python3`` has no ``python`` command, and a PATH ``python`` may be
an interpreter without lerobot; either way ``start`` failed or ran the wrong
environment (#4166). argv[0] is ``sys.executable`` on every builder, the
multi-GPU prefix resolves ``accelerate`` beside that interpreter first, and a
launcher that cannot start is the tool's error envelope, not a traceback.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

# The package re-exports the @tool objects under the modules' own names, so
# ``from strands_robots.tools import lerobot_train`` binds the tool, not the module.
train_mod = importlib.import_module("strands_robots.tools.lerobot_train")
teleop_mod = importlib.import_module("strands_robots.tools.lerobot_teleoperate")


def test_no_launcher_names_a_bare_python() -> None:
    for mod in (train_mod, teleop_mod):
        assert mod.__file__ is not None
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert '["python", "-m"' not in source, f"{mod.__name__} still execs a PATH python"


def test_the_train_command_starts_with_this_interpreter(tmp_path: Path) -> None:
    cmd = train_mod.build_train_command(
        policy_type="act", dataset_root=str(tmp_path), output_dir=str(tmp_path / "out"), num_gpus=1
    )
    assert cmd[:2] == [sys.executable, "-m"]


def test_the_multi_gpu_prefix_resolves_accelerate_beside_the_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "accelerate").write_text("#!/bin/sh\n")
    monkeypatch.setattr(train_mod.sys, "executable", str(fake / "python3.12"))
    assert train_mod._accelerate_launcher() == str(fake / "accelerate")
    (fake / "accelerate").unlink()
    monkeypatch.setattr(train_mod.shutil, "which", lambda name: "/usr/local/bin/accelerate")
    assert train_mod._accelerate_launcher() == "/usr/local/bin/accelerate"
    monkeypatch.setattr(train_mod.shutil, "which", lambda name: None)
    assert train_mod._accelerate_launcher() == "accelerate"


def test_a_launcher_that_cannot_start_is_the_tools_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(train_mod, "session_log_path", lambda name: tmp_path / f"{name}.log")

    def _boom(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "python")

    monkeypatch.setattr(train_mod.subprocess, "Popen", _boom)
    monkeypatch.setattr(
        train_mod, "build_train_command", lambda **kw: ["python", "-m", "lerobot.scripts.lerobot_train"]
    )
    dataset = tmp_path / "ds"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(json.dumps({"total_episodes": 5, "total_tasks": 1}))
    result = train_mod.lerobot_train(
        action="start",
        policy_type="act",
        dataset_root=str(dataset),
        output_dir=str(tmp_path / "out"),
        session_name="t-4166",
    )
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "Could not launch 'python'" in text and "No such file" in text
