"""The judge example records both verdict classes, one frame per applied action.

``examples/19_judge_recorded_episodes.py`` demonstrates filtering a dataset to
its judge-approved successes and training on them. It needs at least one
episode the ``stop_when`` clause ends (success) and one that runs out of steps
(failure); with only failures the filter selects nothing and the training step
is silently skipped.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from strands_robots.tools.episode_judge import load_episode

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "19_judge_recorded_episodes.py"


_load = getattr(load_episode, "__wrapped__", None) or load_episode


def _load_example():
    # A leading digit makes the module unimportable by name; load it by path.
    spec = importlib.util.spec_from_file_location("judge_recorded_episodes", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_judge_example_records_a_success_and_a_failure(tmp_path):
    example = _load_example()
    root = str(tmp_path / "dataset")
    verdicts = example.record_dataset(root, 3)

    assert [v["success"] for v in verdicts] == [False, True, True], verdicts
    # One frame per applied action: an episode the clause stopped holds exactly
    # steps_used frames, none for the state after the firing step.
    lengths = [example._json_payload(_load(root, v["episode"]))["length"] for v in verdicts]
    assert lengths == [v["steps"] for v in verdicts], (lengths, verdicts)
