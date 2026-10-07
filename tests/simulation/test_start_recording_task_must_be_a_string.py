"""Pin: start_recording(task=X) refuses a non-str X up front.

Mirrors _validate_rollout_target on the recorded-task-column path. See
harness issue for the full asymmetric-guard shape.
"""
from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot
from strands_robots.simulation.recording import dataset_recording_task_error


# ---------------------------------------------------------------------------
# Pure helper: shape of the refusal.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        123,
        3.14,
        True,
        False,
        0,
        ["a", "b"],
        {"a": 1},
        (),
        b"bytes",
        None,
    ],
)
def test_dataset_recording_task_error_refuses_non_str(bad):
    err = dataset_recording_task_error("start_recording", bad)
    assert err is not None, f"non-str {type(bad).__name__} must be refused, got None"
    assert err["status"] == "error"
    text = err["content"][0]["text"]
    assert "'task' must be a string" in text
    assert type(bad).__name__ in text, "refusal must name the actual type seen"
    assert "start_recording" in text, "refusal must name the method"


@pytest.mark.parametrize("ok", ["", "pick the cube", "x", "    ", "日本語", "a" * 2048])
def test_dataset_recording_task_error_passes_strings(ok):
    assert dataset_recording_task_error("start_recording", ok) is None


# ---------------------------------------------------------------------------
# End-to-end: SimEngine.start_recording refuses non-str task before any
# dataset-stack probe (so the lerobot extra does not need to be installed
# for the guard to fire).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        123,
        3.14,
        True,
        0,
        ["pick", "cube"],
        {"a": 1},
        None,
    ],
)
def test_start_recording_refuses_non_str_task_upfront(bad, tmp_path):
    sim = Robot("so101", mesh=False)
    r = sim.start_recording(
        repo_id="local/task_guard_test",
        root=str(tmp_path / "d"),
        task=bad,
        fps=30,
        overwrite=True,
    )
    assert r["status"] == "error"
    text = r["content"][0]["text"]
    assert "'task' must be a string" in text, (
        "refusal must name the task kwarg, not the downstream recorder error; "
        f"got: {text[:200]}"
    )


# ---------------------------------------------------------------------------
# Symmetry: the design intent is already codified on run_policy via
# _validate_rollout_target. This test records the sibling pattern so the
# two surfaces cannot drift apart on what a usable task label is.
# ---------------------------------------------------------------------------
def test_task_guard_mirrors_instruction_guard_on_rollout_target():
    """The two helpers refuse the same non-str shapes.

    start_recording(task=X) feeds the parquet task column via
    DatasetRecorder.default_task; run_policy(instruction=X) feeds the
    same column via the per-frame ``task`` kwarg. One rule for one
    column.
    """
    from strands_robots.simulation.base import SimEngine

    for bad in (123, 3.14, True, 0, False, ["a"], {"k": "v"}, (), b"b", None):
        # instruction side
        inst_err = SimEngine._validate_rollout_target(
            robot_name=None, instruction=bad, method="run_policy"
        )
        # task side
        task_err = dataset_recording_task_error("start_recording", bad)
        assert (inst_err is None) == (task_err is None), (
            f"guards disagree on {type(bad).__name__}: "
            f"instruction={inst_err is not None}, task={task_err is not None}"
        )
