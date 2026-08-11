"""Newton recording lifecycle guards: the error, guard, and no-op paths.

The happy path (start -> capture -> save_episode -> parquet) is covered by
``test_dataset_recording``. This module pins the surrounding contracts that a
happy-path recording never exercises:

* ``start_recording`` fails loudly and actionably when the ``lerobot`` extra is
  missing, instead of dead-ending later in dataset creation.
* A recorder-creation failure resets ``recording`` to False (so a subsequent
  attempt is not wedged "recording" with no recorder) and returns an error
  rather than raising past the tool boundary.
* The default ``root=None`` resolves the on-disk dataset dir from ``repo_id``.
* An existing on-disk dataset resumes (appends) instead of recreating.
* The per-frame capture hook is a safe no-op in every state it can be called
  in with nothing to write, so it never raises inside the run-policy loop: an
  unknown robot, no recorder attached, and recording already stopped.

The engine is built through ``__new__`` (as in ``test_dataset_recording``) so
the recording lifecycle runs without the optional Newton/Warp physics stack.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")

import strands_robots.dataset_recorder as dataset_recorder
from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_SO100_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def _make_engine(world: SimWorld) -> NewtonSimEngine:
    """Build a NewtonSimEngine bound to ``world`` without the Warp stack."""
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = object()  # non-None sentinel: "world created"
    engine.default_width = 64
    engine.default_height = 48
    return engine


def _world_with_robot(name: str = "so100") -> SimWorld:
    world = SimWorld()
    world.robots[name] = SimRobot(
        name=name, urdf_path="so100.xml", data_config="so100", joint_names=list(_SO100_JOINTS)
    )
    return world


class TestStartRecordingGuards:
    def test_missing_lerobot_extra_returns_actionable_error(self, monkeypatch, tmp_path):
        # When the lerobot extra is absent, start_recording must not dead-end in
        # dataset creation - it returns an error that names the install extra.
        reason = "lerobot is not installed (ModuleNotFoundError: No module named 'lerobot'). Install lerobot >= 0.6.0 with: pip install 'strands-robots[lerobot]'"
        monkeypatch.setattr(dataset_recorder, "lerobot_dataset_import_error", lambda: reason)
        engine = _make_engine(_world_with_robot())

        result = engine.start_recording(repo_id="local/sim_recording", root=str(tmp_path / "dataset"))

        assert result["status"] == "error"
        text = result["content"][0]["text"]
        # Surfaced verbatim, so the caller sees which dependency is missing.
        assert reason in text
        assert "strands-robots[lerobot]" in text

    def test_no_world_returns_error(self, tmp_path):
        engine = NewtonSimEngine.__new__(NewtonSimEngine)
        engine._world = None
        engine._model = None

        result = engine.start_recording(repo_id="local/sim_recording", root=str(tmp_path / "dataset"))

        assert result["status"] == "error"
        assert "create_world" in result["content"][0]["text"]

    def test_recorder_creation_failure_resets_recording_flag(self, monkeypatch):
        # root=None exercises the repo_id -> HF-cache dir resolution. A failing
        # create() must reset the recording flag (so the world is not wedged in
        # a "recording" state with no recorder) and surface the cause.
        def _boom(**_kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(dataset_recorder.DatasetRecorder, "create", staticmethod(_boom))
        engine = _make_engine(_world_with_robot())

        result = engine.start_recording(repo_id="owner/name", root=None)

        assert result["status"] == "error"
        assert "disk full" in result["content"][0]["text"]
        assert engine._world._backend_state["recording"] is False
        # The dir was resolved into the HF cache tree from the owner/name id.
        assert "owner/name" in engine._world._backend_state["last_dataset_root"]

    def test_existing_dataset_resumes_instead_of_recreating(self, monkeypatch, tmp_path):
        # A dataset dir with a meta/ dir on disk must take the resume (append)
        # branch: DatasetRecorder.resume is called, create() is not.
        dataset_dir = tmp_path / "existing_ds"
        (dataset_dir / "meta").mkdir(parents=True)

        resumed_sentinel = object()
        created = []

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
        engine = _make_engine(_world_with_robot())
        monkeypatch.setattr(engine, "_verify_resume_schema", lambda *a, **k: None)

        result = engine.start_recording(repo_id="owner/name", root=str(dataset_dir))

        assert result["status"] == "success"
        assert engine._world._backend_state["dataset_recorder"] is resumed_sentinel
        assert created == []  # create() must not run on the resume branch


class _RejectingRecorder:
    """Recorder mid-flush: any further frame write is a hard error.

    ``DatasetRecordingMixin.stop_recording`` flips ``recording`` to False and
    only then flushes the trailing episode, leaving the recorder attached
    across ``save_episode()``. A rollout thread whose hook fires inside that
    window holds exactly this object, so the write the flag guard prevents is
    not hypothetical.
    """

    def __init__(self):
        self.add_frame_calls = 0

    def add_frame(self, *_args, **_kwargs):
        self.add_frame_calls += 1
        raise RuntimeError("add_frame after the episode was saved")


class TestRunPolicyHookGuards:
    def test_hook_is_none_for_unknown_robot(self):
        engine = _make_engine(_world_with_robot())
        assert engine._make_run_policy_hook("ghost", "pick") is None

    def test_hook_is_noop_without_recorder(self):
        # recording flagged True but no recorder attached: the hook must return
        # early rather than raise, so a run-policy loop never crashes mid-rollout.
        engine = _make_engine(_world_with_robot())
        hook = engine._make_run_policy_hook("so100", "pick")
        assert hook is not None
        engine._world._backend_state["recording"] = True
        engine._world._backend_state["dataset_recorder"] = None

        obs = {j: 0.0 for j in _SO100_JOINTS}
        action = {j: 0.0 for j in _SO100_JOINTS}
        hook(0, obs, action)  # must not raise

        assert engine._world.robots["so100"].policy_steps == 1

    def test_hook_is_noop_after_recording_stops(self):
        # The state stop_recording leaves while it flushes the trailing episode:
        # the flag is already False and the recorder is still attached. The flag
        # is read first, so the hook must return before touching the recorder.
        engine = _make_engine(_world_with_robot())
        hook = engine._make_run_policy_hook("so100", "pick")
        assert hook is not None
        recorder = _RejectingRecorder()
        engine._world._backend_state["recording"] = False
        engine._world._backend_state["dataset_recorder"] = recorder

        obs = {j: 0.0 for j in _SO100_JOINTS}
        action = {j: 0.0 for j in _SO100_JOINTS}
        hook(4, obs, action)  # must not raise

        assert recorder.add_frame_calls == 0
        # The counter still advances: the hook ran and returned early rather
        # than not having been called at all.
        assert engine._world.robots["so100"].policy_steps == 5
