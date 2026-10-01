"""An RL run leaves its curve, not just its last iteration.

``BaseRLAlgo.train`` (and the FastSAC / FastTD3 overrides) overwrote
``last_metrics`` every iteration, logged nothing per iteration and wrote no
scalars, so ``TrainResult.metrics`` held only the final iteration and whether a
run learned could not be judged after it ended. Every iteration now goes to
``<output_dir>/metrics.jsonl`` and to the log on ``log_interval``.
"""

from __future__ import annotations

import json
import logging
import os

import pytest

from strands_robots.training.base import TrainSpec
from strands_robots.training.rl.base_algo import METRICS_FILENAME, BaseRLAlgo, RLTrainSpec, TrainingHistory


class _Learner(BaseRLAlgo):
    """Reward rises by one per iteration; the loss falls."""

    def __init__(self, fail_at: int | None = None) -> None:
        self._it = 0
        self._fail_at = fail_at

    @property
    def provider_name(self) -> str:
        return "learner"

    def validate(self, spec: TrainSpec) -> list[str]:
        return []

    def setup(self, spec: RLTrainSpec) -> None:
        self.steps_per_iter = 10

    def collect_rollout(self) -> dict[str, float]:
        self._it += 1
        if self._fail_at is not None and self._it == self._fail_at:
            raise RuntimeError("simulator died")
        return {"mean_reward": float(self._it)}

    def update(self) -> dict[str, float]:
        return {"loss": 1.0 / self._it, "grad_norm": float("nan")}

    def save_checkpoint(self, output_dir: str, iteration: int | None = None) -> str:
        return output_dir


def _rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_every_iteration_is_written_and_the_result_points_at_it(tmp_path) -> None:
    result = _Learner().train(RLTrainSpec(total_timesteps=50, output_dir=str(tmp_path), log_interval=2))

    assert result.status == "success"
    path = result.metrics["metrics_path"]
    assert path == os.path.join(str(tmp_path), METRICS_FILENAME)
    rows = _rows(path)
    assert [r["iteration"] for r in rows] == [1, 2, 3, 4, 5]
    assert all(isinstance(r["iteration"], int) for r in rows)
    assert [r["mean_reward"] for r in rows] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert rows[0]["grad_norm"] is None  # non-finite is null, not an invalid JSON NaN
    assert result.metrics["iterations_recorded"] == 5
    assert result.metrics["mean_reward"] == 5.0  # the last iteration stays where it was


def test_a_run_that_dies_still_leaves_its_curve(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="simulator died"):
        _Learner(fail_at=4).train(RLTrainSpec(total_timesteps=50, output_dir=str(tmp_path), log_interval=1))

    assert [r["mean_reward"] for r in _rows(str(tmp_path / METRICS_FILENAME))] == [1.0, 2.0, 3.0]


def test_a_rerun_starts_a_new_curve(tmp_path) -> None:
    spec = RLTrainSpec(total_timesteps=30, output_dir=str(tmp_path), log_interval=1)
    _Learner().train(spec)
    _Learner().train(spec)
    assert len(_rows(str(tmp_path / METRICS_FILENAME))) == 3


def test_the_interval_is_logged(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="strands_robots.training.rl.base_algo")
    _Learner().train(RLTrainSpec(total_timesteps=50, output_dir=str(tmp_path), log_interval=2))
    logged = [r.getMessage() for r in caplog.records if "learner iteration" in r.getMessage()]
    # the first, every second, and the last
    assert [m.split(":")[0] for m in logged] == [
        "learner iteration 1/5",
        "learner iteration 2/5",
        "learner iteration 4/5",
        "learner iteration 5/5",
    ]
    assert "mean_reward=5" in logged[-1]


def test_without_an_output_dir_nothing_is_written() -> None:
    history = TrainingHistory("learner", "", num_iters=1, log_interval=0)
    history.record({"iteration": 1, "mean_reward": 1.0})
    history.close()
    assert history.summary() == {"metrics_path": None, "iterations_recorded": 1}


@pytest.mark.parametrize("module", ["fast_sac", "fast_td3"])
def test_the_off_policy_loops_record_it_too(module) -> None:
    import inspect

    mod = __import__(f"strands_robots.training.rl.{module}", fromlist=["x"])
    source = inspect.getsource(mod)
    assert "TrainingHistory(" in source and "history.record(last_metrics)" in source
