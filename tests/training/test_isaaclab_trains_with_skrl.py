"""The ``isaaclab`` trainer runs skrl as well as rsl_rl.

``extra['rl_library']`` accepted only ``rsl_rl``, so every Isaac Lab task whose
agent configs are skrl's - the AMP humanoids (Walk / Run / Dance), the
multi-agent MAPPO / IPPO configs (Pendulum-MARL, Shadow-Handover), the
Cartpole showcases - could not be trained through strands (ILAC-001). skrl has
no ``--run_name``, writes ``checkpoints/agent_<timestep>.pt`` and reports its
rewards only to TensorBoard; these cells pin how the trainer names, finds and
reads such a run.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from strands_robots.training.isaaclab import (
    SKRL_REWARD_TAG,
    SUPPORTED_RL_LIBRARIES,
    IsaacLabTrainer,
    latest_model,
    parse_skrl_log,
    read_tensorboard_scalars,
)
from tests.training.test_isaaclab import _spec, _trainer, fake_python  # noqa: F401 - fixture

JOB = "isaaclab-20260930-075009-6e09f0824dfb"


def _varint(n: int) -> bytes:
    out = b""
    while True:
        byte, n = n & 0x7F, n >> 7
        out += bytes([byte | (0x80 if n else 0)])
        if not n:
            return out


def _field(num: int, wire: int, payload: bytes) -> bytes:
    return _varint(num << 3 | wire) + payload


def _event(step: int, tag: str, value: float) -> bytes:
    """An ``Event{step, summary{value{tag, simple_value}}}`` in a TFRecord frame (CRCs zeroed)."""
    val = _field(1, 2, _varint(len(tag)) + tag.encode()) + _field(2, 5, struct.pack("<f", value))
    summary = _field(1, 2, _varint(len(val)) + val)
    event = (
        _field(1, 1, struct.pack("<d", 1.0))
        + _field(2, 0, _varint(step))
        + _field(5, 2, _varint(len(summary)) + summary)
    )
    return struct.pack("<Q", len(event)) + b"\0" * 4 + event + b"\0" * 4


@pytest.fixture
def skrl_run(tmp_path: Path) -> Path:
    run = tmp_path / "logs" / "skrl" / "cartpole" / f"2026-09-30_07-50-12_ppo_torch_{JOB}"
    (run / "checkpoints").mkdir(parents=True)
    for t in (32, 320, 64):
        (run / "checkpoints" / f"agent_{t}.pt").write_bytes(b"x")
    (run / "checkpoints" / "best_agent.pt").write_bytes(b"x")
    events = b"".join(_event(s, SKRL_REWARD_TAG, r) for s, r in ((16, -0.3), (160, 2.0), (320, 4.8)))
    events += _event(320, "Loss / Policy loss", 0.1)
    events += b"\x40\x00\x00"  # a torn tail: the run is still writing
    (run / "events.out.tfevents.1790754333.host.1.0").write_bytes(events)
    return run


def test_skrl_is_a_supported_library() -> None:
    assert SUPPORTED_RL_LIBRARIES == ("rsl_rl", "skrl")


def test_the_rewards_are_read_from_tensorboard(skrl_run: Path) -> None:
    assert read_tensorboard_scalars(str(skrl_run), SKRL_REWARD_TAG) == [
        (16, pytest.approx(-0.3)),
        (160, pytest.approx(2.0)),
        (320, pytest.approx(4.8)),
    ]


def test_the_log_and_the_events_give_the_rsl_rl_metric_keys(skrl_run: Path) -> None:
    log = "  0%|  | 0/320 [00:00<?]\r 50%|##  | 160/320 [00:02<00:02]\r100%|####| 320/320 [00:05<00:00]\nTraining time: 9.82 seconds\n"
    m = parse_skrl_log(log, str(skrl_run), max_iterations=20)
    # 320 timesteps over 20 iterations; 0-based like rsl_rl's "Learning iteration 19/20"
    assert m["latest_iteration"] == 19
    assert (m["first_reward"], m["latest_reward"]) == (pytest.approx(-0.3), pytest.approx(4.8))
    assert m["best_iteration"] == 19 and m["learning"] is True and m["training_time_s"] == 9.82


def test_the_newest_agent_checkpoint_is_the_latest_model(skrl_run: Path) -> None:
    latest = latest_model(str(skrl_run))
    assert latest is not None and Path(latest).name == "agent_320.pt"


def test_the_job_id_names_the_run_without_run_name(fake_python, tmp_path: Path) -> None:  # noqa: F811
    cmd = _trainer().build_command(_spec(tmp_path, rl_library="skrl"), JOB)
    assert "--run_name" not in cmd
    assert f"agent.agent.experiment.experiment_name={JOB}" in cmd
    assert cmd[cmd.index("--rl_library") + 1] == "skrl"


def test_rsl_rl_keeps_its_run_name(fake_python, tmp_path: Path) -> None:  # noqa: F811
    cmd = _trainer().build_command(_spec(tmp_path), JOB)
    assert cmd[cmd.index("--run_name") + 1] == JOB
    assert not any(a.startswith("agent.agent.experiment") for a in cmd)


def test_a_learning_rate_lands_on_skrls_field(fake_python, tmp_path: Path) -> None:  # noqa: F811
    spec = _spec(tmp_path, rl_library="skrl")
    spec.learning_rate = 3e-4
    cmd = _trainer().build_command(spec, JOB)
    assert "agent.agent.learning_rate=0.0003" in cmd and "agent.agent.learning_rate_scheduler=null" in cmd
    assert not any(a.startswith("agent.algorithm.") for a in cmd)


def test_save_freq_in_iterations_is_refused_for_skrl(fake_python, tmp_path: Path) -> None:  # noqa: F811
    spec = _spec(tmp_path, steps=100, rl_library="skrl")
    spec.save_freq = 10
    assert any("checkpoint_interval" in p for p in _trainer().validate(spec))


def test_the_run_finds_its_task_from_a_skrl_directory_name(fake_python, skrl_run: Path) -> None:  # noqa: F811
    trainer = _trainer()
    job = trainer._jobs_dir / JOB
    job.mkdir(parents=True)
    (job / "job.json").write_text('{"task": "Isaac-Cartpole"}')
    assert trainer.run_task(skrl_run) == "Isaac-Cartpole"
    assert trainer.latest_checkpoint(str(skrl_run.parents[3]), task="Isaac-Cartpole") == str(skrl_run)


def test_export_says_a_skrl_checkpoint_is_not_convertible_yet(fake_python, skrl_run: Path, tmp_path: Path) -> None:  # noqa: F811
    with pytest.raises(ValueError, match="skrl checkpoint"):
        IsaacLabTrainer().export(None, str(skrl_run))  # type: ignore[arg-type]
