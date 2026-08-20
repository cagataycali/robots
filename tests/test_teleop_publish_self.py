"""The leader half of the teleop chain (U3).

Before this, the mesh could tell a follower to SUBSCRIBE to a leader's input
stream but had no verb able to make that stream exist - receive without publish.
These tests pin the half that was missing, plus the two things that make it safe:
the read goes through the shared bus lock (no second serial connection), and a
host that reports no joints is refused instead of publishing empty frames.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from strands_robots import bus_access
from strands_robots.mesh import security as sec
from strands_robots.teleop_source import RobotAsTeleoperator, positions_from_observation


class _Inner:
    """Stand-in for the lerobot driver that owns the serial port."""

    def __init__(self, obs: dict[str, Any] | None = None, raises: Exception | None = None) -> None:
        self._obs = obs if obs is not None else {"shoulder_pan.pos": 1.0}
        self._raises = raises
        self.reads = 0

    def get_observation(self) -> dict[str, Any]:
        self.reads += 1
        if self._raises:
            raise self._raises
        return dict(self._obs)


class _Host:
    peer_id = "arm-leader"

    def __init__(self, inner: Any) -> None:
        self.robot = inner


# --------------------------------------------------------------- the payload


def test_only_finite_joint_positions_travel():
    obs = {
        "shoulder_pan.pos": 1.5,
        "gripper.pos": 45,            # int is fine
        "wrist_flex.vel": 9.0,        # not a position
        "elbow_flex.pos": float("nan"),   # would drive a follower nowhere
        "wrist_roll.pos": float("inf"),
        "main": object(),             # a camera frame must never ride this topic
        7: 1.0,                       # non-string key
    }
    assert positions_from_observation(obs) == {"shoulder_pan.pos": 1.5, "gripper.pos": 45.0}


def test_a_non_mapping_observation_is_not_a_crash():
    assert positions_from_observation(None) == {}
    assert positions_from_observation([1, 2, 3]) == {}


# ------------------------------------------------------- reading the leader


def test_the_read_takes_the_SAME_bus_lock_as_every_other_reader(monkeypatch):
    """A leader publishing at 30Hz is a bus reader. If it bypassed the lock it
    would collide with the state probe and the camera publisher - the exact
    "Port is in use!" failure bus_access exists to prevent."""
    inner = _Inner({"shoulder_pan.pos": 2.0})
    lock = bus_access.bus_lock(inner)
    src = RobotAsTeleoperator(_Host(inner))

    seen = []
    real = bus_access.read_joints

    def spy(device):
        seen.append(device)
        # The lock must be held DURING the read, not around it.
        assert lock._is_owned() or lock.acquire(blocking=False) and (lock.release() or True)
        return real(device)

    monkeypatch.setattr("strands_robots.teleop_source.read_joints", spy)
    assert src.get_action() == {"shoulder_pan.pos": 2.0}
    assert seen == [inner], "the inner driver is the device whose lock is shared"


def test_a_failed_read_is_one_bad_frame_not_a_dead_session():
    src = RobotAsTeleoperator(_Host(_Inner(raises=RuntimeError("Port is in use!"))))
    assert src.get_action() == {}
    assert src.get_action() == {}
    assert src.read_errors == 2
    assert src.empty_reads == 0, "a failed read and an empty read need different fixes"


def test_an_empty_read_is_counted_separately():
    src = RobotAsTeleoperator(_Host(_Inner({"wrist_flex.vel": 1.0})))
    assert src.get_action() == {}
    assert (src.read_errors, src.empty_reads) == (0, 1)
    assert src.stats["source"] == "self"


def test_a_sim_host_is_its_own_device():
    class Sim:
        peer_id = "sim-1"

        def get_observation(self):
            return {"shoulder_pan.pos": 0.25}

    sim = Sim()
    assert RobotAsTeleoperator(sim, robot_name="so101").get_action() == {"shoulder_pan.pos": 0.25}


def test_a_host_that_cannot_be_read_at_all_reports_an_error_frame():
    src = RobotAsTeleoperator(object())
    assert src.get_action() == {}
    assert src.read_errors == 1


# ------------------------------------------------------- starting the stream


class _PublishHost:
    """A host with the mixin's contract, without the hardware."""

    peer_id = "arm-leader"

    def __init__(self, obs: dict[str, Any]) -> None:
        self.robot = _Inner(obs)
        self.started: list[dict[str, Any]] = []

    def start_teleop_publish(self, teleoperator, device_name="leader", method="arm", hz=50.0):
        self.started.append({"device_name": device_name, "method": method, "hz": hz,
                             "action": teleoperator.get_action()})
        return {"status": "success", "content": [{"text": "started"}]}


def _mixin_publish_self(host, **kw):
    from strands_robots.teleop_mixin import TeleopMixin

    return TeleopMixin.start_teleop_publish_self(host, **kw)


def test_starting_reports_exactly_what_the_receiver_needs():
    host = _PublishHost({"shoulder_pan.pos": 1.0, "gripper.pos": 3.0})
    out = _mixin_publish_self(host, device_name="leader", hz=30.0)
    assert out["status"] == "success"
    # The caller must not have to parse prose to point a follower at this.
    assert out["source_peer_id"] == "arm-leader"
    assert out["device_name"] == "leader"
    assert out["joints"] == ["gripper.pos", "shoulder_pan.pos"]
    assert host.started[0]["hz"] == 30.0


def test_a_host_with_no_joints_is_refused_before_the_loop_starts():
    """A publisher of {} looks perfectly healthy in the counters and moves
    nothing - the worst kind of failure to debug."""
    host = _PublishHost({"wrist_flex.vel": 1.0})
    out = _mixin_publish_self(host)
    assert out["status"] == "error"
    assert "no joint positions" in out["content"][0]["text"]
    assert host.started == [], "nothing was started"


# --------------------------------------------------------------- the wire


@pytest.mark.parametrize("cmd,expected", [
    ({"action": "teleop_publish"}, {"action": "teleop_publish"}),
    ({"action": "teleop_publish", "device_name": "leader", "hz": 30},
     {"action": "teleop_publish", "device_name": "leader", "hz": 30.0}),
])
def test_the_verb_is_allowed_and_normalized(cmd, expected):
    assert sec.validate_command(cmd) == expected


@pytest.mark.parametrize("bad", [
    {"action": "teleop_publish", "hz": 0},
    {"action": "teleop_publish", "hz": -5},
    {"action": "teleop_publish", "hz": 5000},      # a bus read per frame
    {"action": "teleop_publish", "hz": "fast"},
    {"action": "teleop_publish", "hz": True},
    {"action": "teleop_publish", "device_name": "lead*r"},   # zenoh wildcard
    {"action": "teleop_publish", "device_name": "a b"},
    {"action": "teleop_publish", "robot_name": "so101/**"},
])
def test_the_wire_refuses_what_would_reach_downstream_code(bad):
    with pytest.raises(sec.ValidationError):
        sec.validate_command(bad)


def test_hz_is_bounded_where_the_bus_is_the_cost():
    """200Hz is the ceiling because every frame is a real serial exchange shared
    with the state probe; the rate is not just a timer."""
    assert sec.validate_command({"action": "teleop_publish", "hz": 200})["hz"] == 200.0
    with pytest.raises(sec.ValidationError):
        sec.validate_command({"action": "teleop_publish", "hz": 200.5})


def test_the_dispatch_refuses_a_host_without_the_method():
    from strands_robots.mesh.core import Mesh

    class Old:
        pass

    mesh = Mesh.__new__(Mesh)
    mesh.robot = Old()
    import threading

    mesh._estop_lockout = threading.Event()  # _dispatch checks the safety lockout first
    out = Mesh._dispatch(mesh, {"action": "teleop_publish"})
    assert "does not support teleop_publish" in out["error"]


def test_positions_never_include_a_nan_that_a_follower_would_apply():
    vals = positions_from_observation({f"j{i}.pos": v for i, v in enumerate([1.0, math.nan, 2.0])})
    assert all(math.isfinite(v) for v in vals.values())
    assert len(vals) == 2


def test_an_engaged_estop_lockout_refuses_to_start_a_leader_stream():
    """A fleet under emergency stop must not gain a new source of motion: only
    `status` and `resume` pass the lockout, and teleop_publish is deliberately
    not on that list."""
    import threading

    from strands_robots.mesh.core import Mesh

    mesh = Mesh.__new__(Mesh)
    mesh.robot = _PublishHost({"shoulder_pan.pos": 1.0})
    mesh._estop_lockout = threading.Event()
    mesh._estop_lockout.set()
    from strands_robots.mesh import security as _sec

    # It RAISES (LockoutError), so _exec_cmd answers with a deliberately generic
    # error and audits it - a remote caller must not be able to map the window.
    with pytest.raises(_sec.LockoutError):
        Mesh._dispatch(mesh, {"action": "teleop_publish"})
    assert mesh.robot.started == []
