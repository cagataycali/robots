"""Isaac recording lifecycle guards: the error, guard and no-op paths.

The happy path (start -> capture -> save_episode -> parquet) and the
dataset-stack unavailability report are covered by ``test_dataset_recording``.
This module pins the surrounding contracts a happy-path recording never
exercises, mirroring the Newton module of the same name:

* A recorder-creation failure resets ``recording`` to False - so the engine is
  not left wedged "recording" with no recorder - and returns an error rather
  than raising past the tool boundary.
* An existing on-disk dataset resumes (appends) instead of recreating, and the
  create path does not also run.
* The per-step capture hook is a safe no-op in every state it can be called in
  with nothing to write, so it never raises inside the run-policy loop: an
  unknown robot (covered by ``test_dataset_recording``), no recorder attached,
  and recording already stopped.

``IsaacSimulation`` and ``NewtonSimEngine`` each define ``start_recording`` and
``_make_run_policy_hook`` in their own backend mixin - the shared
``DatasetRecordingMixin`` carries the lifecycle around them, not these guards -
so these are independent copies of one contract, and a guard driven on one
backend says nothing about the other.

The engine is built through ``__new__`` (as in ``test_dataset_recording``) so
the recording lifecycle runs without the Isaac Sim Kit runtime. The hook guards
need no optional dependency; the ``start_recording`` guards need the ``lerobot``
extra and are gated below.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState

_SO100_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def _make_engine(robots: dict[str, _RobotState] | None = None) -> IsaacSimulation:
    """Build a skeleton IsaacSimulation without booting the Isaac Kit runtime."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._config = IsaacConfig(render_mode="rtx_realtime")
    engine._lock = threading.RLock()
    engine._world = None
    engine._world_created = True
    engine._robots = robots if robots is not None else {}
    engine._cameras = {}
    engine._objects = {}
    engine._prim_registry = []
    engine._cams_rec_state = None
    engine._recording_state_dict = {}
    engine._action_controllers = {}
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._replicated = False
    engine._num_envs_active = 1
    engine._pump_running = False
    engine._main_tid = threading.get_ident()
    return engine


def _engine_with_robot(name: str = "so100") -> IsaacSimulation:
    return _make_engine(
        {
            name: _RobotState(
                name=name,
                prim_path=f"/World/Robots/{name}",
                joint_names=list(_SO100_JOINTS),
                data_config="so100",
            )
        }
    )


class _RejectingRecorder:
    """Recorder mid-flush: any further frame write is a hard error.

    ``DatasetRecordingMixin.stop_recording`` flips ``recording`` to False and
    only then flushes the trailing episode, leaving the recorder attached
    across ``save_episode()``. A rollout thread whose hook fires inside that
    window holds exactly this object, so the write the flag guard prevents is
    not hypothetical.
    """

    def __init__(self) -> None:
        self.add_frame_calls = 0

    def add_frame(self, *_args: Any, **_kwargs: Any) -> None:
        self.add_frame_calls += 1
        raise RuntimeError("add_frame after the episode was saved")


def _observation() -> dict[str, float]:
    return {joint: 0.0 for joint in _SO100_JOINTS}


class TestRunPolicyHookGuards:
    """The per-step hook's early-outs. No optional dependency required."""

    def test_hook_is_noop_without_recorder(self) -> None:
        # recording flagged True but no recorder attached: the hook must return
        # early rather than raise, so a run-policy loop never crashes mid-rollout.
        engine = _engine_with_robot()
        hook = engine._make_run_policy_hook("so100", "pick")
        assert hook is not None
        state = engine._recording_state()
        assert state is not None
        state["recording"] = True
        state["dataset_recorder"] = None

        hook(0, _observation(), _observation())  # must not raise

        # Isaac's ``_RobotState`` does not declare ``policy_steps`` - the hook
        # sets it dynamically, where Newton's ``SimRobot`` declares the field -
        # so it is read through getattr, and its presence is itself evidence the
        # hook body ran rather than the hook never having been called.
        assert getattr(engine._robots["so100"], "policy_steps", None) == 1

    def test_hook_is_noop_after_recording_stops(self) -> None:
        # The state stop_recording leaves while it flushes the trailing episode:
        # the flag is already False and the recorder is still attached. The flag
        # is read first, so the hook must return before touching the recorder.
        engine = _engine_with_robot()
        hook = engine._make_run_policy_hook("so100", "pick")
        assert hook is not None
        recorder = _RejectingRecorder()
        state = engine._recording_state()
        assert state is not None
        state["recording"] = False
        state["dataset_recorder"] = recorder

        hook(4, _observation(), _observation())  # must not raise

        assert recorder.add_frame_calls == 0
        # The counter still advances: the hook ran and returned early rather
        # than not having been called at all.
        assert getattr(engine._robots["so100"], "policy_steps", None) == 5


# --- start_recording lifecycle (needs the lerobot extra) --------------------

pytest.importorskip("lerobot")

import strands_robots.dataset_recorder as dataset_recorder  # noqa: E402


class TestStartRecordingGuards:
    def test_recorder_creation_failure_resets_recording_flag(self, monkeypatch, tmp_path) -> None:
        # start_recording sets recording=True before creating the recorder, so a
        # failing create() must put it back: otherwise the engine is wedged
        # "recording" with nothing attached and every later frame is dropped by
        # the hook's recorder guard while stop_recording reports no recorder.
        def _boom(**_kwargs: Any) -> None:
            raise RuntimeError("disk full")

        monkeypatch.setattr(dataset_recorder.DatasetRecorder, "create", staticmethod(_boom))
        engine = _engine_with_robot()

        result = engine.start_recording(repo_id="owner/name", root=str(tmp_path / "ds"))

        assert result["status"] == "error"
        # The cause is surfaced rather than raising past the tool boundary.
        assert "disk full" in result["content"][0]["text"]
        state = engine._recording_state()
        assert state is not None
        assert state["recording"] is False
        assert state.get("dataset_recorder") is None

    def test_existing_dataset_resumes_instead_of_recreating(self, monkeypatch, tmp_path) -> None:
        # A dataset dir carrying meta/ must take the resume (append) branch:
        # resume() is called and attached, create() does not run, and the
        # schema is verified against the recorder that came back.
        dataset_dir = tmp_path / "existing_ds"
        (dataset_dir / "meta").mkdir(parents=True)

        resumed_sentinel = object()
        created: list[bool] = []
        verified: list[Any] = []

        monkeypatch.setattr(
            dataset_recorder.DatasetRecorder,
            "resume",
            staticmethod(lambda **_kwargs: resumed_sentinel),
        )
        monkeypatch.setattr(
            dataset_recorder.DatasetRecorder,
            "create",
            staticmethod(lambda **_kwargs: created.append(True)),
        )
        engine = _engine_with_robot()
        monkeypatch.setattr(engine, "_verify_resume_schema", lambda recorder, *a, **k: verified.append(recorder))

        result = engine.start_recording(repo_id="owner/name", root=str(dataset_dir))

        assert result["status"] == "success"
        state = engine._recording_state()
        assert state is not None
        assert state["dataset_recorder"] is resumed_sentinel
        assert created == []  # create() must not run on the resume branch
        assert verified == [resumed_sentinel]  # the resumed schema is checked
