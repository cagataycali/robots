"""Every RL trainer's ``train()`` closes the env its ``setup()`` built.

The trainer owns the env lifecycle - ``spec.env_factory`` hands it a factory
precisely so nothing else holds the instance - but no ``train()`` ever called
``env.close()``. For a single ``SimEnv`` that was invisible (its ``close`` is a
documented no-op), which is exactly why the leak went unnoticed: a vectorized
run builds a ``VecSimEnv`` whose reused ``ThreadPoolExecutor`` is only shut
down by ``close()``, so every vectorized ``train()`` left ``min(num_envs, 8)``
idle worker threads behind for the life of the process while reporting success.

These pin the fix: ``BaseRLAlgo.train`` (PPO inherits it) and the off-policy
overrides (FastSAC, FastTD3) all close in ``finally``, the close is safe on
the validation-failure path (no env exists yet), and the documented
train-then-``evaluate()`` continuation on the same instance still works after
the close because a closed ``VecSimEnv`` steps serially.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from strands_robots.training.rl import (  # noqa: E402
    FastSacTrainer,
    FastTd3Trainer,
    PpoTrainer,
    RLTrainSpec,
    SimEnv,
    VecSimEnv,
)


class _FakeEngine:
    def __init__(self) -> None:
        self._j = 0.0
        self._v = 0.0

    def list_robots(self) -> list[str]:
        return ["fake"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["J"]

    def robot_action_keys(self, robot_name: str) -> list[str]:
        # Duck-typed fake: this robot's one joint is its one actuator, the
        # shape ``SimEnv`` sizes its action head from.
        return ["J"]

    def reset(self) -> dict:
        self._j = 0.0
        self._v = 0.0
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images: bool = False) -> dict:
        return {"J": self._j, "J.vel": self._v}

    def send_action(self, action, robot_name=None, n_substeps: int = 1) -> dict:
        a = float(action[0]) if len(action) else 0.0
        self._v = 0.1 * a
        self._j += self._v
        return {"status": "success"}


class _CloseRecordingEnv(SimEnv):
    """A real ``SimEnv`` that records whether ``close()`` was called.

    ``SimEnv.close`` is a no-op by contract, so the single-env half of the fix
    is observable only through a recorder - which is also what proves the
    trainer calls ``close`` on the *interface*, not on a ``VecSimEnv`` special
    case.
    """

    closed = False

    def close(self) -> None:
        """Record the close, then defer to the no-op contract."""
        self.closed = True
        super().close()


def _make_env():  # type: ignore[no-untyped-def]
    return _CloseRecordingEnv(
        _FakeEngine(),
        actor_obs_keys=["J", "J.vel"],
        reward_terms=[lambda e: -abs(float(e.get_observation(skip_images=True)["J"]) - 0.2)],
        action_dim=1,
        max_episode_steps=8,
    )


def _spec(**overrides: Any) -> RLTrainSpec:
    base: dict[str, Any] = {
        "env_factory": _make_env,
        "output_dir": "/tmp/rl_env_close_tests",
        "total_timesteps": 32,
        "rollout_steps": 8,
        "num_mini_batches": 4,
        "num_learning_epochs": 1,
        "learning_starts": 16,
        "batch_size": 16,
        "hidden_dims": (8,),
        "seed": 0,
    }
    base.update(overrides)
    return RLTrainSpec(**base)


@pytest.mark.parametrize("trainer_cls", [PpoTrainer, FastSacTrainer, FastTd3Trainer])
def test_train_closes_the_single_env(trainer_cls, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """All three train() paths close the env they built, single-env included."""
    trainer = trainer_cls()
    result = trainer.train(_spec(output_dir=str(tmp_path)))
    assert result.status == "success"
    assert trainer.env.closed is True


@pytest.mark.parametrize("trainer_cls", [PpoTrainer, FastTd3Trainer])
def test_train_shuts_down_the_vec_env_pool(trainer_cls, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The vectorized backends leave no live ThreadPoolExecutor behind."""
    trainer = trainer_cls()
    result = trainer.train(_spec(output_dir=str(tmp_path), num_envs=4, total_timesteps=8 * 4 * 2, learning_starts=32))
    assert result.status == "success"
    assert isinstance(trainer.env, VecSimEnv)
    assert trainer.env._executor is None, "train() left the VecSimEnv thread pool running"


@pytest.mark.parametrize("trainer_cls", [PpoTrainer, FastTd3Trainer])
def test_evaluate_still_works_after_the_close(trainer_cls, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The documented train -> evaluate continuation survives the close.

    A closed ``VecSimEnv`` steps serially (its ``_map`` falls back when the
    pool is gone) and ``evaluate`` runs on sub-env 0 anyway, so closing at the
    end of ``train`` must not cost the same-instance eval mode - and
    ``evaluate`` itself never re-closes, so nothing double-closes either.
    """
    trainer = trainer_cls()
    result = trainer.train(_spec(output_dir=str(tmp_path), num_envs=2, total_timesteps=8 * 2 * 2, learning_starts=16))
    assert result.status == "success"
    ev = trainer.evaluate(num_episodes=2)
    assert ev["num_episodes"] == 2
    assert len(ev["returns"]) == 2


@pytest.mark.parametrize("trainer_cls", [PpoTrainer, FastSacTrainer, FastTd3Trainer])
def test_a_failed_validation_has_no_env_to_close(trainer_cls) -> None:  # type: ignore[no-untyped-def]
    """The fail-closed path returns its error without touching an env."""
    trainer = trainer_cls()
    result = trainer.train(_spec(output_dir=""))  # refused by validate
    assert result.status == "error"
    assert getattr(trainer, "env", None) is None


def test_close_is_idempotent_on_the_vec_env() -> None:
    """A caller closing the env it can still reach after train() cannot raise."""
    trainer = FastTd3Trainer()
    result = trainer.train(_spec(output_dir="/tmp/rl_env_close_idem", num_envs=2, total_timesteps=16))
    assert result.status == "success"
    trainer.env.close()  # second close: VecSimEnv guards the missing pool
    trainer.env.close()
