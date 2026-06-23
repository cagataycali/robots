"""Behavior tests for the ``lerobot_train`` agent tool.

The train tool wraps LeRobot's ``lerobot-train`` script behind a single
agent-facing dispatcher with on-disk session tracking, mirroring
``lerobot_teleoperate``. These tests run hardware- and GPU-free by exercising
the pure command builder directly and by substituting fakes for ``subprocess``
and ``psutil`` in the session lifecycle. They pin:

1. The command builder maps each arg set to the correct ``lerobot-train`` argv
   (single-GPU, multi-GPU accelerate, LoRA, held-out val split, resume).
2. The mutual-exclusion / preflight guards fail with clear errors instead of
   launching a doomed run.
3. The session lifecycle (start -> list -> status -> stop) round-trips through
   the persisted session store.
4. Every user-facing ``text`` field is plain ASCII (the project's no-emoji rule).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.lerobot_train as train_mod

build_train_command = train_mod.build_train_command
lerobot_train = train_mod.lerobot_train
SessionManager = train_mod.SessionManager


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _assert_ascii(text: str) -> None:
    offenders = {hex(ord(c)) for c in text if ord(c) > 127}
    assert not offenders, f"non-ASCII characters in tool output: {offenders}"


def _write_dataset(root: Path, total_episodes: int = 10) -> Path:
    """Create a minimal LeRobot v3 dataset stub with meta/info.json."""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": total_episodes}))
    return root


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(train_mod, "SESSION_DIR", session_dir)
    return session_dir


class _FakeProc:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid


# ---------------------------------------------------------------------------
# build_train_command - argv mapping
# ---------------------------------------------------------------------------
def test_build_single_gpu_command_emits_core_flags() -> None:
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        output_dir="/out/act",
        job_name="cube_ft",
        steps=5000,
        batch_size=16,
        save_freq=1000,
        device="cuda",
        dtype="bfloat16",
    )
    assert cmd[:3] == ["python", "-m", "lerobot.scripts.lerobot_train"]
    assert "--dataset.repo_id=local" in cmd
    assert "--dataset.root=/data/cubes" in cmd
    assert "--policy.type=act" in cmd
    assert "--policy.device=cuda" in cmd
    assert "--policy.push_to_hub=false" in cmd
    assert "--output_dir=/out/act" in cmd
    assert "--job_name=cube_ft" in cmd
    assert "--wandb.enable=false" in cmd
    assert "--steps=5000" in cmd
    assert "--batch_size=16" in cmd
    assert "--save_freq=1000" in cmd
    assert "--policy.dtype=bfloat16" in cmd
    # No accelerate prefix on a single-GPU run.
    assert "accelerate" not in cmd


def test_build_multi_gpu_command_prepends_accelerate_launch() -> None:
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        output_dir="/out/act",
        num_gpus=4,
    )
    assert cmd[:2] == ["accelerate", "launch"]
    assert "--multi_gpu" in cmd
    assert "--num_processes=4" in cmd
    assert "--num_machines=1" in cmd
    assert "--mixed_precision=bf16" in cmd
    # The module is launched via -m after the accelerate flags.
    assert cmd[cmd.index("-m") + 1] == "lerobot.scripts.lerobot_train"
    # Core training flags still present downstream.
    assert "--dataset.root=/data/cubes" in cmd
    assert "--policy.type=act" in cmd


def test_build_lora_command_emits_peft_flags() -> None:
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="smolvla",
        output_dir="/out/smolvla",
        lora=True,
        lora_r=32,
        lora_alpha=64,
        lora_target_modules="all-linear",
    )
    assert "--peft.method_type=LORA" in cmd
    assert "--peft.r=32" in cmd
    assert "--peft.lora_alpha=64" in cmd
    assert "--peft.target_modules=all-linear" in cmd


def test_build_expert_only_emits_flag_for_pi_family() -> None:
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="pi05",
        output_dir="/out/pi05",
        train_expert_only=True,
    )
    assert "--policy.train_expert_only=true" in cmd


def test_lora_and_expert_only_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_train_command(
            dataset_root="/data/cubes",
            policy_type="pi05",
            lora=True,
            train_expert_only=True,
        )


def test_expert_only_rejected_for_non_expert_policy() -> None:
    with pytest.raises(ValueError, match="train_expert_only is only valid"):
        build_train_command(
            dataset_root="/data/cubes",
            policy_type="act",
            train_expert_only=True,
        )


def test_val_episodes_reserves_last_n_episodes(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "ds", total_episodes=10)
    cmd = build_train_command(
        dataset_root=str(root),
        policy_type="act",
        output_dir=str(tmp_path / "out"),
        val_episodes=3,
    )
    # 10 total, reserve last 3 -> train on 0..6.
    assert "--dataset.episodes=[0,1,2,3,4,5,6]" in cmd


def test_val_episodes_rejects_reserving_whole_dataset(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "ds", total_episodes=3)
    with pytest.raises(ValueError, match="leaves no training data"):
        build_train_command(
            dataset_root=str(root),
            policy_type="act",
            val_episodes=3,
        )


def test_resume_emits_two_flag_form_when_checkpoint_exists(tmp_path: Path) -> None:
    out = tmp_path / "out"
    ckpt = out / "checkpoints" / "last" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "train_config.json").write_text("{}")

    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        output_dir=str(out),
        resume=True,
    )
    assert f"--config_path={ckpt / 'train_config.json'}" in cmd
    assert "--resume=true" in cmd
    # Resume path ignores the from-scratch flags.
    assert "--dataset.repo_id=local" not in cmd
    assert "--policy.type=act" not in cmd


def test_resume_without_checkpoint_starts_fresh(tmp_path: Path) -> None:
    out = tmp_path / "out"  # no checkpoints dir
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        output_dir=str(out),
        resume=True,
    )
    # No resumable checkpoint -> normal fresh-run flags, no --config_path.
    assert not any(c.startswith("--config_path=") for c in cmd)
    assert "--dataset.repo_id=local" in cmd
    assert "--policy.type=act" in cmd


def test_extra_flags_passthrough_normalizes_leading_dashes() -> None:
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        output_dir="/out",
        extra_flags={"policy.optimizer_lr": "1e-4", "--num_workers": 8},
    )
    assert "--policy.optimizer_lr=1e-4" in cmd
    assert "--num_workers=8" in cmd


def test_push_to_hub_true_sets_flag() -> None:
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        output_dir="/out",
        push_to_hub=True,
    )
    assert "--policy.push_to_hub=true" in cmd


def test_num_gpus_zero_rejected() -> None:
    with pytest.raises(ValueError, match="num_gpus must be >= 1"):
        build_train_command(dataset_root="/data/cubes", num_gpus=0)


# ---------------------------------------------------------------------------
# Preflight (start action)
# ---------------------------------------------------------------------------
def test_start_errors_when_dataset_missing_info(tmp_path: Path) -> None:
    # Directory exists but has no meta/info.json.
    empty = tmp_path / "empty"
    empty.mkdir()
    result = lerobot_train(action="start", dataset_root=str(empty))
    assert result["status"] == "error"
    text = _texts(result)
    assert "info.json" in text
    _assert_ascii(text)


# ---------------------------------------------------------------------------
# Session lifecycle: start -> list -> status -> stop
# ---------------------------------------------------------------------------
def test_session_lifecycle_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_dataset(tmp_path / "ds", total_episodes=8)

    # Fake Popen so no real training process spawns.
    def _fake_popen(cmd, **kwargs):  # noqa: ANN001
        return _FakeProc(pid=9999)

    monkeypatch.setattr(train_mod.subprocess, "Popen", _fake_popen)
    # Report the fake PID as a live, running process.
    monkeypatch.setattr(train_mod.psutil, "pid_exists", lambda pid: True)

    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def is_running(self) -> bool:
            return True

    monkeypatch.setattr(train_mod.psutil, "Process", _FakeProcess)

    start = lerobot_train(
        action="start",
        dataset_root=str(root),
        policy_type="act",
        output_dir=str(tmp_path / "out"),
        session_name="t1",
    )
    assert start["status"] == "success"
    assert start["session_name"] == "t1"
    assert start["pid"] == 9999
    _assert_ascii(_texts(start))

    listed = lerobot_train(action="list", dataset_root=str(root))
    assert listed["status"] == "success"
    assert "t1" in listed["sessions"]
    _assert_ascii(_texts(listed))

    status = lerobot_train(action="status", dataset_root=str(root), session_name="t1")
    assert status["status"] == "success"
    assert status["is_running"] is True
    assert status["pid"] == 9999
    _assert_ascii(_texts(status))

    # Stop: capture the kill calls instead of touching a real process.
    killed: list[int] = []
    monkeypatch.setattr(train_mod.os, "kill", lambda pid, sig: killed.append(pid))
    # After SIGTERM the process is gone.
    monkeypatch.setattr(train_mod.psutil, "pid_exists", lambda pid: False)

    stop = lerobot_train(action="stop", dataset_root=str(root), session_name="t1")
    assert stop["status"] == "success"
    assert 9999 in killed
    _assert_ascii(_texts(stop))

    # Session store no longer tracks it.
    gone = lerobot_train(action="status", dataset_root=str(root), session_name="t1")
    assert gone["status"] == "error"


def test_status_requires_session_name(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "ds")
    result = lerobot_train(action="status", dataset_root=str(root))
    assert result["status"] == "error"
    _assert_ascii(_texts(result))


def test_unknown_action_errors(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "ds")
    result = lerobot_train(action="frobnicate", dataset_root=str(root))
    assert result["status"] == "error"
    assert "Unknown action" in _texts(result)
