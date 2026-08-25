"""A scene whose cameras do not have distinct dataset column names is refused.

A LeRobot feature name cannot contain ``/``, so a camera's namespace separator
is recorded as ``__`` (:func:`strands_robots.utils.camera_schema_key`). That
mapping is not injective: ``arm/wrist`` and ``arm__wrist`` are two cameras in
the scene and one ``observation.images.arm__wrist`` column in the dataset.

Nothing downstream could tell which camera the column was declared for, and the
three ways of asking each failed differently: recording every camera was refused
far from the cause (as a repeated ``camera_keys`` entry, which reads as the
caller naming one camera twice); ``cameras=`` naming both reported success and
dropped one, leaving the column carrying the other camera's frames; and
``cameras=`` naming one succeeded with the column's contents decided by which
spelling was used. Every one of those happened after ``overwrite=True`` had
already wiped the dataset being replaced.

Covers:

* every one of the three ways of asking is refused, with a message naming both
  cameras and the key they share;
* the refusal happens before any dataset is created, resumed or wiped;
* a scene whose camera names stay distinct after the collapse still records end
  to end (the control), including a namespaced robot camera on its own;
* ``camera_schema_key`` is idempotent and is the single owner of the collapse;
* every backend that declares a dataset schema consults the guard.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import pytest

from strands_robots.simulation.recording import camera_schema_key_collision_error
from strands_robots.utils import camera_schema_key

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

os.environ.setdefault("MUJOCO_GL", "egl")

#: A robot whose MJCF declares its own camera, so the compiled scene namespaces
#: it as ``<robot>/wrist`` and its dataset column is ``<robot>__wrist``.
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


def _scene_cameras(sim) -> list[str]:
    """The compiled scene's camera names, in the model's own order."""
    import mujoco as mj

    model = sim._world._model
    return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]


def _sim(tmp_path, second_camera: str | None, width: int = 128, height: int = 96):
    """A one-joint arm with a namespaced ``arm/wrist`` camera.

    Args:
        tmp_path: Where the robot MJCF is written.
        second_camera: A camera name to add beside the robot's, or ``None``.
        width: Render width of that second camera.
        height: Render height of that second camera.
    """
    from strands_robots.simulation import Simulation

    path = tmp_path / "test_arm.xml"
    path.write_text(_ROBOT_XML)
    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", urdf_path=str(path))
    if second_camera is not None:
        added = sim.add_camera(
            name=second_camera,
            position=[0.0, -0.9, 0.5],
            target=[0.0, 0.0, 0.1],
            width=width,
            height=height,
        )
        assert added["status"] == "success", added
    return sim


def _text(result) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


class TestThePremise:
    """The colliding pair really is two cameras with one column name."""

    def test_the_two_names_are_distinct_cameras_in_the_compiled_scene(self, tmp_path):
        sim = _sim(tmp_path, "arm__wrist")
        try:
            names = _scene_cameras(sim)
            assert "arm/wrist" in names, names
            assert "arm__wrist" in names, names
        finally:
            sim.destroy()

    def test_the_two_names_share_one_column(self):
        assert camera_schema_key("arm/wrist") == camera_schema_key("arm__wrist") == "arm__wrist"

    def test_the_pair_would_be_asked_for_as_distinct_names(self):
        # name_list_error refuses a literally repeated name; these two are
        # distinct, so only their keys collide and it passes them through. That
        # is why the collapse needs its own check rather than a stricter list.
        from strands_robots.utils import name_list_error

        assert name_list_error(["arm/wrist", "arm__wrist"], "cameras", "start_recording") is None

    def test_the_key_is_idempotent(self):
        # This is why a caller may name a camera in either form: applying the
        # collapse to a key returns the key.
        for name in ("arm/wrist", "arm__wrist", "wrist", "a/b/c"):
            assert camera_schema_key(camera_schema_key(name)) == camera_schema_key(name)

    def test_the_camera_receivers_scan_can_see_a_respelling(self):
        # Without this the rule above passes because the pattern never matches
        # anything, rather than because nothing respells it.
        found = [
            ast.unparse(node.func.value)
            for node in ast.walk(ast.parse('cam_name.replace("/", "__")'))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "replace"
            and [getattr(a, "value", None) for a in node.args] == ["/", "__"]
        ]
        assert found == ["cam_name"], found


class TestEveryBackendConsultsTheGuard:
    """A backend that declares a dataset schema declares it from the collapse.

    Derived from the packages that ship a ``start_recording``, so a backend
    added later is graded on arrival rather than being outside a list written
    today.
    """

    @staticmethod
    def _recording_modules() -> dict[str, str]:
        import strands_robots.simulation as simulation

        root = Path(inspect.getsourcefile(simulation)).parent
        found = {}
        for path in sorted(root.glob("*/recording.py")):
            source = path.read_text()
            if "def start_recording" in source:
                found[path.parent.name] = source
        return found

    def test_more_than_one_backend_is_graded(self):
        # Without this the rules below pass on an empty inventory.
        assert len(self._recording_modules()) >= 2, sorted(self._recording_modules())

    def test_every_backend_calls_the_guard_from_start_recording(self):
        missing = []
        for backend, source in self._recording_modules().items():
            tree = ast.parse(source)
            start = next(
                node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "start_recording"
            )
            calls = {
                node.func.id
                for node in ast.walk(start)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            if "camera_schema_key_collision_error" not in calls:
                missing.append(backend)
        assert missing == [], missing

    def test_the_guard_runs_before_the_dataset_target_is_prepared(self):
        # _prepare_dataset_target wipes on overwrite, so a guard called after it
        # refuses a call that has already cost the caller their dataset.
        for backend, source in self._recording_modules().items():
            guard = source.index("camera_schema_key_collision_error(")
            wipe = source.index("_prepare_dataset_target(")
            assert guard < wipe, backend


class TestEveryWayOfRecordingACollidingSceneIsRefused:
    """The ambiguity belongs to the scene, not to one way of recording it."""

    @pytest.mark.parametrize(
        "cameras",
        [
            None,
            ["arm/wrist", "arm__wrist"],
            ["arm__wrist"],
            ["arm/wrist"],
        ],
        ids=["every-camera", "both-spellings", "safe-spelling-only", "raw-spelling-only"],
    )
    def test_start_recording_is_refused(self, tmp_path, cameras):
        sim = _sim(tmp_path, "arm__wrist")
        try:
            kwargs = {} if cameras is None else {"cameras": cameras}
            result = sim.start_recording(
                repo_id="local/collision",
                root=str(tmp_path / "ds"),
                overwrite=True,
                **kwargs,
            )
            assert result["status"] == "error", result
            message = _text(result)
            assert "do not have distinct dataset feature names" in message, message
            # Both cameras and the key they share are named, so the caller can
            # act without reproducing the collapse by hand.
            assert "'arm/wrist'" in message, message
            assert "'arm__wrist'" in message, message
        finally:
            sim.destroy()

    def test_the_refusal_writes_nothing_to_disk(self, tmp_path):
        # overwrite=True wipes the dataset being replaced, so a refusal that
        # happened after the wipe would cost the caller what was already there.
        root = tmp_path / "ds"
        root.mkdir()
        (root / "meta").mkdir()
        (root / "meta" / "info.json").write_text('{"pre": "existing"}')
        sim = _sim(tmp_path, "arm__wrist")
        try:
            result = sim.start_recording(repo_id="local/collision", root=str(root), overwrite=True)
            assert result["status"] == "error", result
            assert (root / "meta" / "info.json").read_text() == '{"pre": "existing"}'
        finally:
            sim.destroy()


class TestNoPartialStateEitherWay:
    """Holds before and after the fix, so it pins the property not the change.

    The pre-fix refusal came from ``DatasetRecorder.create`` and already
    unwound the session, so this is not evidence for the guard - it is the
    invariant the guard must not break by refusing earlier.
    """

    def test_no_recorder_is_left_registered(self, tmp_path):
        sim = _sim(tmp_path, "arm__wrist")
        try:
            sim.start_recording(repo_id="local/collision", root=str(tmp_path / "ds"), overwrite=True)
            assert sim._world._backend_state.get("dataset_recorder") is None
            assert not sim._world._backend_state.get("recording")
        finally:
            sim.destroy()


class TestADistinctSceneStillRecords:
    """Control: the collapse is only refused when it loses information."""

    def test_a_namespaced_robot_camera_records_under_its_collapsed_key(self, tmp_path):
        sim = _sim(tmp_path, None)
        try:
            result = sim.start_recording(
                repo_id="local/ok",
                root=str(tmp_path / "ds"),
                overwrite=True,
                cameras=["arm/wrist"],
            )
            assert result["status"] == "success", _text(result)
            recorder = sim._world._backend_state["dataset_recorder"]
            declared = {
                key[len("observation.images.") :]
                for key in recorder.dataset.features
                if key.startswith("observation.images.")
            }
            assert declared == {"arm__wrist"}, declared
        finally:
            sim.destroy()

    def test_two_cameras_that_stay_distinct_record_a_column_each(self, tmp_path):
        sim = _sim(tmp_path, "overview", width=64, height=64)
        try:
            result = sim.start_recording(
                repo_id="local/ok2",
                root=str(tmp_path / "ds"),
                overwrite=True,
                cameras=["arm/wrist", "overview"],
            )
            assert result["status"] == "success", _text(result)
            recorder = sim._world._backend_state["dataset_recorder"]
            declared = {
                key[len("observation.images.") :]
                for key in recorder.dataset.features
                if key.startswith("observation.images.")
            }
            assert declared == {"arm__wrist", "overview"}, declared
        finally:
            sim.destroy()

    def test_an_unnamed_camera_is_not_a_collision(self):
        # A backend skips an unnamed camera when it declares the schema, so
        # blank names are not columns and two of them do not collide. Without
        # the skip they would all share the key "" and refuse every scene that
        # has more than one.
        assert camera_schema_key_collision_error("start_recording", ["", "", "arm/wrist"]) is None


class TestTheCollapseHasOneOwner:
    """One mapping, so the guard and the schema cannot disagree."""

    def test_no_other_module_collapses_a_camera_name_itself(self):
        # Other domains legitimately map "/" to "__" for their own reasons (an
        # HF repo id becoming a local checkpoint directory, for one), so the
        # claim is about the CAMERA collapse: every receiver whose name says it
        # holds a camera goes through the owner instead of respelling the rule.
        import strands_robots

        package = Path(inspect.getsourcefile(strands_robots)).parent
        owner = Path(inspect.getsourcefile(camera_schema_key)).resolve()
        respellings = []
        for path in sorted(package.rglob("*.py")):
            if path.resolve() == owner:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "replace"
                    and [getattr(a, "value", None) for a in node.args] == ["/", "__"]
                ):
                    receiver = ast.unparse(node.func.value)
                    if "cam" in receiver.lower():
                        respellings.append(f"{path.relative_to(package)}: {receiver}")
        assert respellings == [], respellings


class TestTheRefusalMessage:
    """What the refusal tells a caller, read straight off the domain."""

    def test_the_message_names_every_colliding_group(self):
        # Two independent collisions in one scene are both reported, so a caller
        # renaming one camera is not sent back for a second refusal they could have
        # been told about the first time.
        result = camera_schema_key_collision_error(
            "start_recording",
            ["a/x", "a__x", "b/y", "b__y", "lone"],
        )
        assert result is not None
        message = " ".join(block["text"] for block in result["content"])
        assert "'a__x'" in message and "'b__y'" in message, message
        assert "lone" not in message, message

    def test_a_scene_with_no_cameras_is_not_a_collision(self):
        assert camera_schema_key_collision_error("start_recording", []) is None

    def test_the_method_name_prefixes_the_message(self):
        result = camera_schema_key_collision_error("some_method", ["a/x", "a__x"])
        assert result is not None
        assert result["content"][0]["text"].startswith("some_method: ")
