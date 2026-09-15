"""``start_recording``'s reply names the cameras the dataset records.

``"6 joints, 1 cameras @ 10fps"`` told an agent how many image columns the
dataset had but not which. A replayed agent read it, asked ``render`` for
``top_camera`` - the name it assumed the recording used - and got "not found.
Available: ['default']". The reply that creates the camera columns now lists
their keys, and says plainly when there are none (and why: ``cameras=[]`` or a
camera-less scene) instead of leaving ``0 cameras`` to be skimmed past. One
helper in ``strands_robots.simulation.recording`` serves all three backends.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.recording import recorded_cameras_line

JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]


class TestTheLineNamesTheCameras:
    def test_one_camera_is_named_not_counted(self) -> None:
        line = recorded_cameras_line(JOINTS, ["default"], None, 10)
        assert line == "6 joints, 1 camera ['default'] @ 10fps\n"

    def test_several_cameras_keep_dataset_key_order(self) -> None:
        line = recorded_cameras_line(JOINTS, ["arm__wrist", "top"], ["arm__wrist", "top"], 30)
        assert "2 cameras ['arm__wrist', 'top'] @ 30fps" in line
        assert "No cameras" not in line

    def test_a_named_camera_adds_no_second_line(self) -> None:
        assert recorded_cameras_line(JOINTS, ["default"], None, 10).count("\n") == 1


class TestNoCameraSaysWhatTheDatasetCarries:
    def test_cameras_empty_list_is_named_as_the_cause(self) -> None:
        line = recorded_cameras_line(JOINTS, [], [], 10)
        assert line.startswith("6 joints, 0 cameras [] @ 10fps\n")
        assert "cameras=[] scoped them all out" in line
        assert "joint state and actions only" in line
        assert "no observation.images.*" in line

    def test_a_camera_less_scene_points_at_add_camera(self) -> None:
        line = recorded_cameras_line(JOINTS, [], None, 10)
        assert "No cameras in the scene" in line
        assert "add_camera(...) before start_recording" in line
        assert "cameras=[]" not in line


@pytest.mark.parametrize(
    "module",
    [
        "strands_robots.simulation.mujoco.recording",
        "strands_robots.simulation.isaac.recording",
        "strands_robots.simulation.newton.recording",
    ],
)
def test_every_backend_builds_the_line_from_the_shared_helper(module: str) -> None:
    import importlib
    import inspect

    try:
        mod = importlib.import_module(module)
    except ImportError as e:  # pragma: no cover - optional backend deps
        pytest.skip(f"{module}: {e}")
    src = inspect.getsource(mod)
    assert "recorded_cameras_line(joint_names, camera_keys, cameras, fps)" in src
    assert "cameras @ {fps}fps" not in src


def _mujoco_sim(tool_name: str):
    pytest.importorskip("mujoco")
    pytest.importorskip("lerobot")
    from strands_robots.simulation.mujoco.simulation import Simulation

    return Simulation(tool_name=tool_name, mesh=False)


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
            scoped = sim.stop_recording()
            assert scoped["status"] in {"success", "error"}
        finally:
            sim.cleanup()

    def test_scoping_every_camera_out_is_said_in_the_reply(self, tmp_path) -> None:
        sim = _mujoco_sim("names_cams")
        try:
            sim.create_world()
            assert sim.add_robot(name="arm", data_config="so101")["status"] == "success"
            result = sim.start_recording(repo_id="t/names_cams", root=str(tmp_path), fps=10, cameras=[])
            text = result["content"][0]["text"]
            assert result["status"] == "success", text
            assert "0 cameras [] @ 10fps" in text
            assert "cameras=[] scoped them all out" in text
        finally:
            sim.cleanup()
