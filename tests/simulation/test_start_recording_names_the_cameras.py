"""``start_recording``'s reply names the cameras by a name that answers for them.

``"6 joints, 1 cameras @ 10fps"`` told an agent how many image columns the
dataset had but not which. A replayed agent read it, asked ``render`` for
``top_camera`` - the name it assumed the recording used - and got "not found.
Available: ['default']".

Naming the dataset's COLUMN keys instead reproduces that refusal one namespace
deeper: :func:`~strands_robots.utils.camera_schema_key` collapses the ``/`` of a
robot-scoped camera to ``__`` for the dataset schema, so a reply listing
``so101__wrist`` names a camera every render surface only answers for as
``so101/wrist``. The reply therefore lists the SCENE name - what ``render`` and
``cameras=`` take - and names the dataset column beside it when they differ.
When no camera is recorded, the reason is read off the scene: three causes,
three different remedies.

One helper in ``strands_robots.simulation.recording`` serves all three backends.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.recording import recorded_cameras_line
from tests.simulation.mujoco._gl_probe import requires_gl

JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]


class TestTheLineNamesTheCameras:
    def test_one_camera_is_named_not_counted(self) -> None:
        line = recorded_cameras_line(JOINTS, {"default": "default"}, ["default"], None, 10)
        assert line == "6 joints, 1 camera ['default'] @ 10fps\n"

    def test_several_cameras_keep_dataset_column_order(self) -> None:
        line = recorded_cameras_line(JOINTS, {"top": "top", "side": "side"}, ["top", "side"], ["top", "side"], 30)
        assert line == "6 joints, 2 cameras ['top', 'side'] @ 30fps\n"

    def test_a_camera_whose_column_matches_adds_no_second_line(self) -> None:
        assert recorded_cameras_line(JOINTS, {"default": "default"}, ["default"], None, 10).count("\n") == 1


class TestARenamedColumnIsNamedBesideTheSceneName:
    """The scene name leads; the ``__`` column is stated, never substituted."""

    def test_the_scene_name_is_listed_not_the_column_key(self) -> None:
        line = recorded_cameras_line(JOINTS, {"so101/wrist": "so101__wrist"}, ["so101/wrist"], None, 10)
        assert line.startswith("6 joints, 1 camera ['so101/wrist'] @ 10fps\n")
        assert "['so101__wrist']" not in line

    def test_the_dataset_column_is_named_too(self) -> None:
        line = recorded_cameras_line(JOINTS, {"so101/wrist": "so101__wrist"}, ["so101/wrist"], None, 10)
        assert "'so101/wrist' -> observation.images.so101__wrist" in line
        assert "render and cameras= take the scene name above" in line

    def test_only_the_renamed_cameras_are_spelled_out(self) -> None:
        line = recorded_cameras_line(
            JOINTS,
            {"default": "default", "so101/wrist": "so101__wrist"},
            ["default", "so101/wrist"],
            None,
            10,
        )
        assert line.count("observation.images.") == 1
        assert "observation.images.default" not in line


class TestNoCameraSaysWhatTheDatasetCarriesAndWhy:
    def test_a_camera_less_scene_points_at_add_camera(self) -> None:
        line = recorded_cameras_line(JOINTS, {}, [], None, 10)
        assert line.startswith("6 joints, 0 cameras [] @ 10fps\n")
        assert "No cameras in the scene" in line
        assert "add_camera(...) before start_recording" in line
        assert "joint state and actions only" in line
        assert "no observation.images.*" in line

    def test_a_scoped_out_selection_names_what_it_scoped_out(self) -> None:
        line = recorded_cameras_line(JOINTS, {}, ["default", "so101/wrist"], [], 10)
        assert "cameras= scoped out ['default', 'so101/wrist']" in line
        assert "Omit cameras= or name the ones to keep" in line
        assert "add_camera" not in line

    def test_cameras_that_produce_no_frame_are_not_blamed_on_the_caller(self) -> None:
        """Isaac's ``render_mode='headless'``: cameras exist, frames do not.

        Neither ``add_camera(...)`` nor a different ``cameras=`` would help, so
        neither is offered - the reply says the cameras produce no frame.
        """
        line = recorded_cameras_line(JOINTS, {}, ["top"], None, 10)
        assert "the scene's camera(s) ['top'] produce no frame to record" in line
        assert "add_camera" not in line
        assert "Omit cameras=" not in line

    def test_an_empty_selection_on_a_camera_less_scene_blames_the_scene(self) -> None:
        """``cameras=[]`` scoped nothing out when there was nothing to scope."""
        line = recorded_cameras_line(JOINTS, {}, [], [], 10)
        assert "No cameras in the scene" in line
        assert "scoped out" not in line


@pytest.mark.parametrize(
    "module",
    [
        "strands_robots.simulation.mujoco.recording",
        "strands_robots.simulation.isaac.recording",
        "strands_robots.simulation.newton.recording",
    ],
)
def test_every_backend_builds_the_line_from_the_shared_helper(module: str) -> None:
    import inspect

    # Isaac and Newton are optional extras; importorskip is the house answer for
    # a module that may not be installed (a try/except that skips leaves the name
    # bound only on the success path).
    mod = pytest.importorskip(module)
    src = inspect.getsource(mod)
    assert "recorded_cameras_line(joint_names, recorded_cameras, " in src
    assert "cameras @ {fps}fps" not in src
    # The dataset column keys never reach the reply as the camera's name.
    assert "recorded_cameras_line(joint_names, camera_keys" not in src


def _mujoco_sim(tool_name: str):
    pytest.importorskip("mujoco")
    pytest.importorskip("lerobot")
    from strands_robots.simulation.mujoco.simulation import Simulation

    return Simulation(tool_name=tool_name, mesh=False)


def _schema_line(text: str) -> str:
    return next(line for line in text.splitlines() if " joints, " in line)


class TestOnTheMuJoCoBackend:
    def test_the_reply_names_the_default_camera_the_agent_had_to_guess(self, tmp_path) -> None:
        sim = _mujoco_sim("names_cams")
        try:
            sim.create_world()
            assert sim.add_robot(name="arm", data_config="so101")["status"] == "success"
            result = sim.start_recording(repo_id="t/names_cams", root=str(tmp_path), fps=10)
            text = result["content"][0]["text"]
            assert result["status"] == "success", text
            assert "1 camera ['default'] @ 10fps" in text
            sim.stop_recording()
        finally:
            sim.cleanup()

    @requires_gl
    def test_every_camera_the_reply_names_is_one_render_accepts(self, tmp_path) -> None:
        """The round trip the defect broke: read the reply, render what it named.

        A robot-scoped camera is the case where the dataset column and the scene
        name diverge, so a reply built from the column keys hands back
        ``so101__wrist`` and ``render`` answers "not found. Available:
        ['default', 'so101/wrist']" - the same refusal the reply exists to
        prevent.
        """
        sim = _mujoco_sim("names_cams")
        try:
            sim.create_world()
            assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
            added = sim.add_camera(
                name="so101/wrist",
                parent_body="so101/gripper",
                position=[0.05, 0.0, 0.02],
                target=[0.3, 0.0, 0.0],
            )
            assert added["status"] == "success", added["content"][0]["text"]
            result = sim.start_recording(repo_id="t/named_cams_rt", root=str(tmp_path), fps=10)
            text = result["content"][0]["text"]
            assert result["status"] == "success", text

            import re

            named = re.findall(r"'([^']+)'", _schema_line(text))
            assert "so101/wrist" in named, text
            for name in named:
                rendered = sim.render(camera_name=name)
                assert rendered["status"] == "success", f"{name}: {rendered['content'][0]['text']}"

            # The column that name records to is stated, not substituted for it.
            assert "'so101/wrist' -> observation.images.so101__wrist" in text
            sim.stop_recording()
        finally:
            sim.cleanup()

    def test_scoping_every_camera_out_names_the_cameras_it_scoped_out(self, tmp_path) -> None:
        sim = _mujoco_sim("names_cams")
        try:
            sim.create_world()
            assert sim.add_robot(name="arm", data_config="so101")["status"] == "success"
            result = sim.start_recording(repo_id="t/names_cams", root=str(tmp_path), fps=10, cameras=[])
            text = result["content"][0]["text"]
            assert result["status"] == "success", text
            assert "0 cameras [] @ 10fps" in text
            assert "cameras= scoped out ['default']" in text
            sim.stop_recording()
        finally:
            sim.cleanup()

    @requires_gl
    def test_the_caller_s_column_key_spelling_is_answered_with_the_scene_name(self, tmp_path) -> None:
        """``cameras=`` takes either spelling; the reply always answers in one."""
        sim = _mujoco_sim("names_cams")
        try:
            sim.create_world()
            assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
            sim.add_camera(
                name="so101/wrist",
                parent_body="so101/gripper",
                position=[0.05, 0.0, 0.02],
                target=[0.3, 0.0, 0.0],
            )
            result = sim.start_recording(
                repo_id="t/names_cams_scoped",
                root=str(tmp_path),
                fps=10,
                cameras=["so101__wrist"],
            )
            text = result["content"][0]["text"]
            assert result["status"] == "success", text
            assert "1 camera ['so101/wrist'] @ 10fps" in text
            assert sim.render(camera_name="so101/wrist")["status"] == "success"
            sim.stop_recording()
        finally:
            sim.cleanup()
