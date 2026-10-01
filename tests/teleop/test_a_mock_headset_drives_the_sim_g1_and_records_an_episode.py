"""A mock headset drives the simulated G1 through the whole-body seam and an episode records.

The design (``docs/project/design-wholebody-teleop.md``) claims that a
:class:`~strands_robots.teleop.WholeBodyTeleoperator` is a lerobot-shaped input
device, so the follower's existing ``teleoperate()`` loop drives it unchanged,
and that a recording made through it is a LeRobot v3 dataset the verifier
accepts. Both claims are cheap to prove without hardware or model weights, so
they are proved here, on the MuJoCo G1: a scripted source, an identity joint map,
the 29-joint layout.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pytest

from strands_robots import Robot
from strands_robots.teleop import (
    G1_SIM_JOINTS,
    GRIPPERS,
    ISAACLAB_TO_HARDWARE,
    JointMapRetarget,
    MockPoseSource,
    PoseFrame,
    RetargetOut,
    WholeBodyTeleoperator,
    list_layouts,
    teleop_layout,
)

#: Loop rate the follower's teleop loop runs at, the blog's recording rate.
HZ = 50.0
#: Long enough for tens of frames, short enough for CI.
DURATION_S = 1.5
#: How many of the ``HZ * DURATION_S`` frames the loop must actually deliver.
MIN_FRAMES = 60
#: Physics steps per control tick when the proof drives the sim by hand: 10 x 2 ms = one 50 Hz frame.
SUBSTEPS_PER_TICK = 10
#: Control ticks in the recording proof: 100 x 20 ms = 2 s of sim time = 100 frames at 50 fps.
RECORD_TICKS = 100


@pytest.fixture
def g1() -> Any:
    """A MuJoCo G1 built the way a user builds it, torn down after the test."""
    robot = Robot("g1")
    try:
        yield robot
    finally:
        robot.cleanup()


def _identity_map() -> JointMapRetarget:
    return JointMapRetarget({name: name for name in G1_SIM_JOINTS})


def test_the_mock_source_drives_the_sim_g1_through_the_existing_teleop_loop(g1: Any) -> None:
    """``attach_teleop`` + ``teleoperate`` accept the whole-body device and run it at rate."""
    source = MockPoseSource(period_s=2.0)
    device = WholeBodyTeleoperator(source, _identity_map(), layout="g1_joint_29")

    assert set(device.action_features) == set(G1_SIM_JOINTS)
    g1.attach_teleop(device, name="mock")
    result = g1.teleoperate(hz=HZ, duration=DURATION_S, block=True)
    assert result["status"] == "success", result

    stats = g1.get_teleoperate_status()["content"][1]["json"]
    assert stats["frames"] >= MIN_FRAMES, stats
    assert stats["errors"] == 0, stats
    assert stats["slew_rejected"] == 0, stats
    assert device.frames_used == stats["frames"]
    assert device.frames_stale == 0

    # The follower was told the script's arm targets, and only those: the
    # 29-joint layout has no gripper columns, so the mock's gripper sweep never
    # reaches a robot that has no gripper actuator.
    assert set(device.last_joint_targets) == set(source.scripted_joints(0.0))
    observation = g1.get_observation("g1", skip_images=True)
    moved = [name for name in device.last_joint_targets if abs(observation[name]) > 1e-3]
    assert moved, "the arms never left zero"


def test_a_recording_made_through_the_seam_verifies_as_one_episode(g1: Any, tmp_path: Path) -> None:
    """Source -> retarget -> sim -> recorder -> verify, at the blog's 50 fps."""
    device = WholeBodyTeleoperator(MockPoseSource(period_s=2.0), _identity_map(), layout="g1_joint_29")
    started = g1.start_recording(
        repo_id="local/wholebody_proof", task="wave both arms", fps=int(HZ), root=str(tmp_path)
    )
    assert started["status"] == "success", started

    device.connect()
    for _ in range(RECORD_TICKS):
        action = device.get_action()
        assert action, "the source went stale"
        sent = g1.send_action(action, robot_name="g1")
        assert sent["status"] == "success", sent
        g1.step(SUBSTEPS_PER_TICK)
    device.disconnect()

    stopped = g1.stop_recording()
    assert stopped["status"] == "success", stopped
    verified = g1.verify_dataset_episodes(1)
    assert verified["status"] == "success", verified

    info = json.loads(next(tmp_path.rglob("info.json")).read_text())
    assert info["fps"] == int(HZ)
    assert info["codebase_version"].startswith("v3")
    assert info["features"]["action"]["shape"] == [len(G1_SIM_JOINTS)]
    assert info["features"]["action"]["names"] == list(G1_SIM_JOINTS)
    assert info["total_frames"] >= RECORD_TICKS


def test_the_blog_layout_spells_the_dataset_columns_exactly() -> None:
    """``blog_31_66`` is the ``nepyope/can_clean_final`` feature table, name for name."""
    layout = teleop_layout("blog_31_66")
    assert layout.state_names[:3] == ("kLeftHipPitch.q", "kLeftHipRoll.q", "kLeftHipYaw.q")
    assert layout.state_names[-3:] == ("kRightWristYaw.q", "left_gripper", "right_gripper")
    assert len(layout.state_names) == 31
    assert layout.action_names[0] == "motion_token_0"
    assert layout.action_names[63] == "motion_token_63"
    assert layout.action_names[-2:] == GRIPPERS
    assert len(layout.action_names) == 66
    assert layout.is_token

    joints = teleop_layout("blog_31_31")
    assert joints.action_names == layout.state_names
    assert not joints.is_token
    assert set(list_layouts()) >= {"g1_joint_29", "blog_31_66", "blog_31_31", "lerobot_token_64"}


def test_the_wire_order_permutation_is_a_permutation_with_the_wrists_last() -> None:
    """The breadth-first table maps every hardware joint once and puts the six wrists at 23..28."""
    assert sorted(ISAACLAB_TO_HARDWARE) == list(range(29))
    tail = [G1_SIM_JOINTS[i] for i in ISAACLAB_TO_HARDWARE[23:]]
    assert all("wrist" in name for name in tail), tail


class _FixedEncoder:
    """An encoder that answers one constant token, so the layout can be graded without weights."""

    dim = 64

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def encode(self, out: RetargetOut) -> list[float]:
        del out
        return [float(i) / 64.0 for i in range(64)]


def test_an_encoder_makes_the_action_the_66_blog_columns() -> None:
    """With an encoder the action dict is the token plus grippers, and nothing else."""
    source = MockPoseSource(period_s=2.0)
    encoder = _FixedEncoder()
    device = WholeBodyTeleoperator(source, _identity_map(), encoder=encoder, layout="blog_31_66")
    device.connect()
    assert encoder.resets == 1
    action = device.get_action()
    assert list(action) == list(teleop_layout("blog_31_66").action_names)
    assert action["motion_token_63"] == pytest.approx(63.0 / 64.0)
    assert 0.0 <= action["left_gripper"] <= 1.0
    # The follower-facing joint targets are still there for a sim to follow.
    assert set(device.last_joint_targets) == set(source.scripted_joints(0.0))


def test_encoder_and_layout_must_agree() -> None:
    """A token layout without an encoder, or an encoder with a joint layout, is refused at build time."""
    with pytest.raises(ValueError, match="wants an encoder"):
        WholeBodyTeleoperator(MockPoseSource(), _identity_map(), layout="blog_31_66")
    with pytest.raises(ValueError, match="wants no encoder"):
        WholeBodyTeleoperator(MockPoseSource(), _identity_map(), encoder=_FixedEncoder(), layout="g1_joint_29")
    with pytest.raises(ValueError, match="Unknown teleop layout"):
        teleop_layout("blog_66_31")


def test_a_stale_frame_sends_nothing() -> None:
    """A source whose last frame is older than the budget yields an empty action, so the follower holds."""

    class Frozen:
        name = "frozen"
        is_connected = True

        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

        def latest(self) -> PoseFrame:
            return PoseFrame(t_mono=time.monotonic() - 1.0, frame_index=1, joints={"left_elbow_joint": 0.1})

    device = WholeBodyTeleoperator(Frozen(), _identity_map(), layout="g1_joint_29", stale_after_s=0.2)
    assert device.get_action() == {}
    assert device.frames_stale == 1
    assert device.frames_used == 0


def test_the_joint_map_drops_unmapped_axes_and_refuses_a_collision() -> None:
    """A leader's extra axis never reaches the follower; two sources on one joint is a build error."""
    retarget = JointMapRetarget({"exo_elbow": ("left_elbow_joint", -1.0, math.pi / 2)})
    frame = PoseFrame(t_mono=time.monotonic(), frame_index=1, joints={"exo_elbow": 0.5, "joystick_x": 0.9})
    out = retarget(frame)
    assert out.joints == {"left_elbow_joint": pytest.approx(-0.5 + math.pi / 2)}
    with pytest.raises(ValueError, match="several source joints"):
        JointMapRetarget({"a": "left_elbow_joint", "b": "left_elbow_joint"})
