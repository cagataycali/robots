"""The ``isaaclab`` trainer runs rl_games: Factory, Forge and AutoMate.

Eight Isaac Lab tasks register rl_games agent configs only - Factory
PegInsert / GearMesh / NutThread, Forge, AutoMate Assembly / Disassembly - so
with rsl_rl and skrl alone they stayed out of strands' reach (ILAC-001, 2/2).
rl_games names its run ``full_experiment_name``, counts epochs, saves
``nn/last_<cfg>_ep_<epoch>_rew__<reward>_.pth`` and logs rewards to
TensorBoard under ``summaries/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots.training.isaaclab import (
    RL_GAMES_REWARD_TAG,
    RL_GAMES_SUCCESS_TAG,
    SUPPORTED_RL_LIBRARIES,
    IsaacLabTrainer,
    latest_model,
    parse_rl_games_log,
)
from tests.training.test_isaaclab import _spec, _trainer, fake_python  # noqa: F401 - fixture
from tests.training.test_isaaclab_trains_with_skrl import _event

JOB = "isaaclab-20260930-080320-cc84c0ccbc38"


@pytest.fixture
def run(tmp_path: Path) -> Path:
    d = tmp_path / "logs" / "rl_games" / "Factory" / f"2026-09-30_08-03-20_{JOB}"
    (d / "nn").mkdir(parents=True)
    (d / "summaries").mkdir()
    for ep, rew in ((3, 40.79), (5, 49.78)):
        (d / "nn" / f"last_Factory_ep_{ep}_rew__{rew}_.pth").write_bytes(b"x")
    (d / "nn" / "Factory.pth").write_bytes(b"x")  # the best, not an epoch
    events = b"".join(_event(s, RL_GAMES_REWARD_TAG, r) for s, r in ((1, 37.9), (3, 41.0), (5, 45.2)))
    events += _event(5, RL_GAMES_SUCCESS_TAG, 0.25)
    (d / "summaries" / "events.out.tfevents.1790755191.host").write_bytes(events)
    return d


def test_rl_games_is_supported() -> None:
    assert "rl_games" in SUPPORTED_RL_LIBRARIES


def test_epochs_and_rewards_read_like_an_rsl_rl_run(run: Path) -> None:
    log = "fps step: 585 epoch: 1/5 frames: 0\nfps step: 546 epoch: 5/5 frames: 32768\nTraining time: 179.46 seconds\n"
    m = parse_rl_games_log(log, str(run))
    assert m["latest_iteration"] == 4  # 0-based: the 5th of 5
    assert (m["first_reward"], m["latest_reward"]) == (pytest.approx(37.9), pytest.approx(45.2))
    assert m["best_iteration"] == 4 and m["learning"] is True
    assert m["success_rate"] == {"latest": 0.25, "max": 0.25}
    assert m["training_time_s"] == 179.46


def test_the_newest_epoch_checkpoint_is_the_latest_model(run: Path) -> None:
    assert Path(latest_model(str(run))).name == "last_Factory_ep_5_rew__49.78_.pth"


def test_the_job_id_names_the_run_with_its_start_time(fake_python, tmp_path: Path) -> None:  # noqa: F811
    cmd = _trainer().build_command(_spec(tmp_path, rl_library="rl_games"), JOB)
    assert "--run_name" not in cmd
    assert f"agent.params.config.full_experiment_name=2026-09-30_08-03-20_{JOB}" in cmd


def test_learning_rate_and_save_freq_land_on_rl_games_fields(fake_python, tmp_path: Path) -> None:  # noqa: F811
    spec = _spec(tmp_path, steps=100, rl_library="rl_games")
    spec.learning_rate, spec.save_freq = 1e-4, 10
    assert _trainer().validate(spec) == []
    cmd = _trainer().build_command(spec, JOB)
    assert "agent.params.config.learning_rate=0.0001" in cmd
    assert "agent.params.config.lr_schedule=identity" in cmd
    assert "agent.params.config.save_frequency=10" in cmd
    assert not any(a.startswith(("agent.algorithm.", "agent.save_interval")) for a in cmd)


def test_the_run_is_found_by_task_and_start_time(fake_python, run: Path) -> None:  # noqa: F811
    trainer = _trainer()
    (trainer._jobs_dir / JOB).mkdir(parents=True)
    (trainer._jobs_dir / JOB / "job.json").write_text('{"task": "IsaacContrib-Factory-PegInsert-Direct"}')
    assert trainer.latest_checkpoint(str(run.parents[3]), task="IsaacContrib-Factory-PegInsert-Direct") == str(run)


def test_export_names_an_rl_games_checkpoint_as_not_convertible_yet(run: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint of rl_games"):
        IsaacLabTrainer().export(None, str(run))  # type: ignore[arg-type]
