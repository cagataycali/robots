"""A refused ``start_recording`` leaves the caller's dataset where it was.

``overwrite=True`` is the one posture that deletes a real LeRobotDataset without
asking (:meth:`~strands_robots.simulation.recording.DatasetRecordingMixin._prepare_dataset_target`),
and it is what the ``run_policy`` agent tool records with. Every camera-shaped
refusal ``start_recording`` makes was already ahead of that deletion - the
``cameras=`` name-list domain, the boolean postures, the fps domain, and the
scene-level key collision, whose own guard is documented as running "before any
dataset is created, resumed or wiped". One was not: a single unknown name in
``cameras=`` was refused from inside the schema-declaration block, roughly a
hundred lines after the wipe, so the call reported an error having already
removed the dataset it was replacing - and the refusal's remedy ("Add them with
``add_camera(...)`` ... or omit ``cameras=``") asks for a retry against the data
that same call destroyed.

Nothing between resolving the target and building the recorder reads or writes
the dataset directory - the schema is read from the scene - so the deletion is
deferred to the last statement before the recorder is built, which puts every
refusal ahead of it.

Covers:

* an existing dataset survives a refused ``cameras=`` call, byte for byte, and
  still reopens through ``LeRobotDataset`` with its frames and image column;
* the refusal itself is unchanged - same status, same message, no session left
  open (the invariant refusing earlier must not break);
* no refusal sits between the deletion and the recorder in any backend that
  ships a ``start_recording``, derived from the tree so a backend or a refusal
  added later is graded on arrival;
* ``overwrite=True`` still replaces an existing dataset and a non-dataset
  directory is still not clobbered - the deletion is deferred, not dropped.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

os.environ.setdefault("MUJOCO_GL", "egl")

_ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <camera name="wrist" pos="0.25 0 0.05" xyaxes="0 -1 0 0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
  </actuator>
</mujoco>
"""

_UNKNOWN = "wrsit"  # the transposition an operator actually types for "wrist"


def _sim(tmp_path):
    """A one-joint arm carrying a single ``arm/wrist`` camera."""
    from strands_robots.simulation import Simulation

    path = tmp_path / "test_arm.xml"
    path.write_text(_ROBOT_XML)
    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", urdf_path=str(path))
    return sim


def _text(result) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


def _marker_dataset(root: Path) -> str:
    """Stand in for "a dataset already lives here", byte-addressably.

    ``_prepare_dataset_target`` decides resume-vs-create on the presence of
    ``meta/``, and the property under test is whether the bytes already on disk
    are still there afterwards - so a marker file states it exactly, and the
    round-trip cell below carries the same claim for a real recording.
    """
    (root / "meta").mkdir(parents=True)
    payload = '{"pre": "existing"}'
    (root / "meta" / "info.json").write_text(payload)
    return payload


def _record_one_episode(sim, root: Path, *, cameras: list[str] | None = None) -> None:
    """Write a real single-episode LeRobotDataset at ``root``."""
    started = sim.start_recording(
        repo_id="local/refused_recording",
        root=str(root),
        task="pan the arm",
        overwrite=True,
        cameras=cameras,
    )
    assert started["status"] == "success", _text(started)
    rollout = sim.run_policy(robot_name="arm", policy_provider="mock", n_steps=6, control_frequency=30.0)
    assert rollout["status"] == "success", _text(rollout)
    stopped = sim.stop_recording()
    assert stopped["status"] == "success", _text(stopped)


def _episode_count(root: Path) -> int:
    return int(json.loads((root / "meta" / "info.json").read_text())["total_episodes"])


class TestThePremise:
    """The name really is unknown, and naming it really is refused."""

    def test_the_scene_does_not_carry_the_requested_camera(self, tmp_path):
        import mujoco as mj

        sim = _sim(tmp_path)
        try:
            model = sim._world._model
            names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
            assert "arm/wrist" in names, names
            assert _UNKNOWN not in names, names
        finally:
            sim.destroy()

    def test_the_refusal_names_the_unknown_camera_and_what_exists(self, tmp_path):
        # Holds either way: the refusal is right, and this pins that deferring
        # the deletion did not change what the caller is told.
        sim = _sim(tmp_path)
        try:
            result = sim.start_recording(
                repo_id="local/refused_recording",
                root=str(tmp_path / "ds"),
                overwrite=True,
                cameras=["arm/wrist", _UNKNOWN],
            )
            assert result["status"] == "error", result
            text = _text(result)
            assert _UNKNOWN in text, text
            # The available list is the scene's RAW names, which is what a
            # caller must pass to add_camera - not the collapsed column name.
            assert "arm/wrist" in text, text
        finally:
            sim.destroy()


class TestARefusedCallKeepsTheDataset:
    """The refusal is not what was wrong - deleting first was."""

    @pytest.mark.parametrize("overwrite", [True, False])
    def test_the_bytes_already_on_disk_are_untouched(self, tmp_path, overwrite):
        # ``overwrite=False`` resumes rather than deletes, so that row held
        # before this change too: it is the control that locates the damage in
        # the destructive posture specifically - the one ``run_policy`` uses.
        root = tmp_path / "ds"
        payload = _marker_dataset(root)
        sim = _sim(tmp_path)
        try:
            result = sim.start_recording(
                repo_id="local/refused_recording",
                root=str(root),
                overwrite=overwrite,
                cameras=["arm/wrist", _UNKNOWN],
            )
            assert result["status"] == "error", result
            assert (root / "meta" / "info.json").exists(), sorted(p.name for p in root.rglob("*"))
            assert (root / "meta" / "info.json").read_text() == payload
        finally:
            sim.destroy()

    def test_a_recorded_dataset_still_reopens_after_a_refused_call(self, tmp_path):
        # The marker cell states the property exactly; this one states it about
        # a dataset LeRobot itself wrote and can still read - parquet, metadata
        # and the per-camera video - which is what the caller stood to lose.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = tmp_path / "ds"
        sim = _sim(tmp_path)
        try:
            _record_one_episode(sim, root, cameras=["arm/wrist"])
            before = LeRobotDataset(repo_id="local/refused_recording", root=str(root))
            frames, episodes = before.meta.total_frames, before.meta.total_episodes
            assert frames > 0 and episodes == 1, (frames, episodes)

            refused = sim.start_recording(
                repo_id="local/refused_recording",
                root=str(root),
                overwrite=True,
                cameras=["arm/wrist", _UNKNOWN],
            )
            assert refused["status"] == "error", _text(refused)
        finally:
            sim.destroy()

        # Asserted before reopening: a deleted local dataset sends
        # ``LeRobotDataset`` to the Hub for ``local/refused_recording``, and the
        # 404 that comes back names neither the directory nor the refusal.
        assert (root / "meta" / "info.json").exists(), f"the refused call removed the dataset at {root}"
        after = LeRobotDataset(repo_id="local/refused_recording", root=str(root))
        assert (after.meta.total_frames, after.meta.total_episodes) == (frames, episodes)
        assert "observation.images.arm__wrist" in after.meta.features, sorted(after.meta.features)
        videos = sorted(path for path in root.rglob("*.mp4"))
        assert videos, sorted(p.name for p in root.rglob("*"))
        assert all(path.stat().st_size > 0 for path in videos), videos

    def test_no_session_is_left_open(self, tmp_path):
        # Holds either way. The refusal already unwound the session flag; this
        # pins that refusing before the deletion does not leave one behind.
        root = tmp_path / "ds"
        _marker_dataset(root)
        sim = _sim(tmp_path)
        try:
            sim.start_recording(
                repo_id="local/refused_recording",
                root=str(root),
                overwrite=True,
                cameras=["arm/wrist", _UNKNOWN],
            )
            assert not sim._world._backend_state.get("recording")
            assert sim._world._backend_state.get("dataset_recorder") is None
        finally:
            sim.destroy()


class TestTheDeletionIsDeferredNotDropped:
    """Controls: the four documented outcomes still happen, just later."""

    def test_overwrite_still_replaces_an_existing_dataset(self, tmp_path):
        # The failure mode of "stop deleting so early" is "stop deleting": a
        # second overwrite=True rollout must replace the first episode, not
        # append to it.
        root = tmp_path / "ds"
        sim = _sim(tmp_path)
        try:
            _record_one_episode(sim, root, cameras=["arm/wrist"])
            assert _episode_count(root) == 1
            _record_one_episode(sim, root, cameras=["arm/wrist"])
            assert _episode_count(root) == 1, "overwrite=True appended instead of replacing"
        finally:
            sim.destroy()

    def test_a_non_dataset_directory_is_still_not_clobbered(self, tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        (root / "unrelated.txt").write_text("keep me")
        sim = _sim(tmp_path)
        try:
            result = sim.start_recording(
                repo_id="local/refused_recording",
                root=str(root),
                overwrite=False,
                cameras=["arm/wrist"],
            )
            assert result["status"] == "error", result
            assert (root / "unrelated.txt").read_text() == "keep me"
        finally:
            sim.destroy()


class TestNoRefusalFollowsTheDeletion:
    """Derived from the tree, so a later backend or refusal is graded on arrival.

    The rule is positional because that is what the defect was: the guard was
    written, correct, and reached too late. Only the span between the deletion
    and the recorder is constrained - the two returns after the recorder is
    built are its success envelope and its own failure handler, neither of which
    could precede the object they report on.
    """

    @staticmethod
    def _recording_modules() -> dict[str, str]:
        import strands_robots.simulation as simulation

        root = Path(inspect.getfile(simulation)).parent
        return {
            path.parent.name: path.read_text()
            for path in sorted(root.glob("*/recording.py"))
            if "def start_recording" in path.read_text()
        }

    @staticmethod
    def _span(source: str) -> tuple[int, int, list[int]]:
        """``(deletion lineno, recorder lineno, refusal linenos between them)``."""
        start = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "start_recording"
        )
        deletion = next(
            node.lineno
            for node in ast.walk(start)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_prepare_dataset_target"
        )
        recorder = min(
            node.lineno
            for node in ast.walk(start)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"create", "resume"}
            and isinstance(node.func.value, ast.Name)
            and "DatasetRecorder" in node.func.value.id
        )
        between = [
            node.lineno
            for node in ast.walk(start)
            if isinstance(node, ast.Return) and deletion < node.lineno < recorder
        ]
        return deletion, recorder, between

    def test_more_than_one_backend_is_graded(self):
        # Without this the rules below pass on an empty inventory.
        assert len(self._recording_modules()) >= 2, sorted(self._recording_modules())

    def test_every_backend_still_resolves_the_target_before_building_the_recorder(self):
        for backend, source in self._recording_modules().items():
            deletion, recorder, _ = self._span(source)
            assert deletion < recorder, backend

    def test_no_backend_refuses_after_the_deletion(self):
        offenders = {
            backend: between
            for backend, source in self._recording_modules().items()
            if (between := self._span(source)[2])
        }
        assert offenders == {}, offenders

    def test_every_backend_still_carries_the_camera_scoping_refusal(self):
        # Non-vacuity for the rule above, and it holds either way: deleting the
        # guard would also satisfy "no refusal after the deletion", so the rule
        # is only meaningful while the guard it moved is still there.
        for backend, source in self._recording_modules().items():
            assert "unknown camera(s)" in source, backend
