"""Regression: positional action-vector binding uses actuators, not joints.

A ``LeRobotDataset``'s ``action`` column is written in the robot's *actuator*
order (``SimEngine.robot_action_keys``); see the recorder in
``strands_robots/simulation/mujoco/recording.py``, which deliberately keys the
action schema by actuators because keying by joint names "records all-zero
action columns" for robots whose actuators differ from their joints.

On replay the recorded vector must be mapped back onto the SAME actuator keys.
Mapping it onto ``robot_joint_names`` instead (the pre-fix behaviour) mis-maps
and silently drops commanded DOFs on any robot whose actuator set differs from
its joint set - passive/mimic finger joints or a tendon-driven gripper - while
``replay_episode`` still reports ``status="success"``: a silent record->replay
round-trip corruption. The same convention governs ``send_action`` when given a
raw numeric action vector (``SimEngine._coerce_action``).

``aloha`` is the decisive case: 14 actuators (per-arm 6 joints + 1 gripper
tendon) vs 16 joints (each arm adds 2 passive finger joints with no driving
actuator). A 14-long action vector recorded in actuator order was previously
mapped onto the 16-joint list, shifting the entire right arm by one index and
dropping both grippers.
"""

from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402


@pytest.fixture
def aloha_sim():
    s = Simulation()
    s.create_world(ground_plane=True)
    s.add_robot("aloha")
    yield s
    s.cleanup()


def test_aloha_actuators_differ_from_joints(aloha_sim):
    """Precondition: aloha has passive finger joints -> actuators != joints.

    If this ever stops holding the round-trip test below is vacuous, so pin it.
    """
    joints = aloha_sim.robot_joint_names("aloha")
    actuators = aloha_sim.robot_action_keys("aloha")
    assert joints != actuators
    assert len(actuators) == 14
    assert len(joints) == 16
    # The grippers are tendon actuators with no matching joint name; the finger
    # joints are passive (no driving actuator).
    assert "left/gripper" in actuators and "right/gripper" in actuators
    assert "left/gripper" not in joints and "right/gripper" not in joints


class TestSendActionVectorBindsToActuators:
    """``send_action`` numeric vector binds to actuator order, not joint order."""

    def test_actuator_length_vector_applies(self, aloha_sim):
        """A vector of length == actuator count applies to every actuator."""
        n = len(aloha_sim.robot_action_keys("aloha"))  # 14
        result = aloha_sim.send_action([0.0] * n, robot_name="aloha", n_substeps=1)
        assert result["status"] == "success", result
        # No key was left unresolved (no silent drops).
        assert not any(isinstance(b, dict) and b.get("json", {}).get("unresolved_keys") for b in result["content"])

    def test_joint_length_vector_is_rejected(self, aloha_sim):
        """A vector of length == joint count no longer matches the actuator count."""
        n_joints = len(aloha_sim.robot_joint_names("aloha"))  # 16
        result = aloha_sim.send_action([0.0] * n_joints, robot_name="aloha")
        assert result["status"] == "error"
        assert "action-key count 14" in result["content"][0]["text"]


class TestReplayRoundTrip:
    """A self-recorded dataset replays onto the actuators it was recorded from."""

    def test_replay_maps_recorded_actions_to_actuators(self, aloha_sim):
        pytest.importorskip("lerobot")
        from strands_robots.dataset_recorder import load_lerobot_episode

        root = tempfile.mkdtemp(prefix="aloha_replay_rt_")
        repo = "local/aloha_replay_rt"

        # Record a short episode through the real recorder (no cameras -> fast,
        # action-only). The recorder writes the action column in actuator order.
        assert aloha_sim.start_recording(repo_id=repo, task="rt", fps=30, root=root, cameras=[])["status"] == "success"
        assert (
            aloha_sim.run_policy(
                robot_name="aloha", policy_provider="mock", n_steps=6, control_frequency=30, fast_mode=True
            )["status"]
            == "success"
        )
        assert aloha_sim.stop_recording()["status"] == "success"

        ds, _start, ep_len = load_lerobot_episode(repo, 0, root)
        action_column_names = list(ds.meta.features["action"]["names"])
        assert action_column_names == aloha_sim.robot_action_keys("aloha")

        # Capture the action dicts replay emits to send_action.
        emitted: list[list[str]] = []
        original = aloha_sim.send_action

        def spy(action, robot_name=None, n_substeps=1):
            if isinstance(action, dict):
                emitted.append(list(action.keys()))
            return original(action, robot_name=robot_name, n_substeps=n_substeps)

        aloha_sim.send_action = spy  # type: ignore[method-assign]
        try:
            rep = PolicyRunner(aloha_sim).replay(repo, robot_name="aloha", root=root, speed=1000.0)
        finally:
            aloha_sim.send_action = original  # type: ignore[method-assign]

        assert rep["status"] == "success"
        assert len(emitted) == ep_len
        # Every replayed frame must map the recorded vector back onto the exact
        # actuator keys the recorder wrote (positional round-trip). Pre-fix this
        # bound to robot_joint_names, shifting the right arm and dropping the
        # grippers (which have no matching joint name).
        for keys in emitted:
            assert keys == action_column_names
