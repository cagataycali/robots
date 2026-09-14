# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: a teleop ``robot_name`` the host cannot route to is refused.

``robot_name`` is consumed only *inside* the teleop loop -
``send_action(merged, robot_name=...)`` on every tick, and inside the closure
:meth:`~strands_robots.teleop_mixin.TeleopMixin.start_teleop_receive` installs -
which is the category :meth:`~strands_robots.teleop_mixin.TeleopMixin.teleoperate`
already grades at the door for ``hz`` and ``duration``: *"an unusable value used
to be reported as a started session and only misbehave on the background
thread"*. ``robot_name`` was the member of that category still ungraded.

Measured on the pre-fix tree against a real MuJoCo world holding ``so100`` and
``arm2``, driving one leader at 50 Hz with ``robot_name="arm-2"``:

=========================================  ===================  ==================
call                                       before               after
=========================================  ===================  ==================
``teleoperate(block=True, duration=0.6)``  30 frames/30 errors  refused at the door
leader ``connect()`` calls for that call    1                    0
``teleoperate()`` (background) at start    ``success``          ``error``
``start_teleop_receive`` at that key       replaced a live one  live one survives
=========================================  ===================  ==================

The follower moved in neither row (``max_abs_delta`` 0.0 on both arms): the
session was a silent no-op, which is the outcome
:meth:`~strands_robots.teleop_mixin.TeleopMixin._teleop_stats` names in its own
comment as the one its status derivation exists to refuse. That derivation runs
at *stop*, so background mode - the default - reported a started session and said
nothing until the operator asked for it back.

The receive door is the sharper half. Its two mesh identifiers are validated
*ahead of* the teardown of any stream already registered under that key,
precisely so "a refused call cannot stop a live one"; ``robot_name`` skipped that
guarantee, so a typo stopped the receiver that was following this leader and
installed one whose every frame the world refused - under ``status="success"``.

The refusal itself is not new text: the host answers with the
close-match message its own ``send_action`` produced on each of those frames
("Robot 'arm-2' not found. Did you mean: arm2? ..."), so the door and the loop
speak with one voice.

Deliberately unchanged, and pinned below: ``robot_name=None`` in a multi-robot
world is still accepted (the loop resolves it), and a host that wraps exactly one
device still accepts and ignores any ``robot_name`` - the documented hardware
parity, which is what the mixin's default hook preserves.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

from strands_robots.teleop_mixin import TeleopMixin

#: The name an operator meant to type. Close enough that the refusal's
#: close-match hint must offer the real one.
TYPO = "arm-2"
TARGET = "arm2"


class _Leader:
    """Minimal lerobot-Teleoperator-shaped double that records its connects."""

    def __init__(self, action: dict[str, float]) -> None:
        self._action = action
        self.name = "fake_leader"
        self.id = None
        self.is_connected = False
        self.connect_calls = 0

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 - lerobot signature
        self.is_connected = True
        self.connect_calls += 1

    def disconnect(self) -> None:
        self.is_connected = False

    def get_action(self) -> dict[str, float]:
        return dict(self._action)


class _Mesh:
    """Enough mesh for the receive door: alive, a peer id, a subscription token."""

    alive = True
    peer_id = "follower-1"

    def __init__(self) -> None:
        self._estop_lockout = None

    def subscribe(self, *a: Any, **k: Any) -> str:
        return "sub"

    def unsubscribe(self, *a: Any, **k: Any) -> None:
        return None


class _SingleDeviceHost(TeleopMixin):
    """A host shaped like the hardware ``Robot``: one device, no world to search."""

    def __init__(self) -> None:
        self.tool_name_str = "single"
        self.mesh = None
        self.peer_id = None
        self.sent: list[str | None] = []

    def send_action(self, action: dict[str, float], robot_name: str | None = None) -> dict[str, Any]:  # noqa: ARG002
        self.sent.append(robot_name)
        return {"status": "success", "content": [{"text": "ok"}]}


@pytest.fixture
def sim() -> Iterator[Any]:
    """A real MuJoCo world holding two robots, so a name can be wrong."""
    pytest.importorskip("mujoco")
    from strands_robots.simulation import create_simulation
    from strands_robots.simulation.model_registry import resolve_model_path

    model = resolve_model_path("so100")
    if model is None:  # pragma: no cover - asset not cached on this host
        pytest.skip("so100 asset is not available")
    engine = create_simulation("mujoco")
    try:
        engine.create_world()
        engine.add_robot(name="so100", urdf_path=str(model))
        engine.add_robot(name=TARGET, urdf_path=str(model), position=[0.6, 0.0, 0.0])
        assert TARGET in engine.list_robots()
        yield engine
    finally:
        engine.cleanup()


def _leader(sim: Any) -> _Leader:
    return _Leader(_command(sim))


def _command(sim: Any) -> dict[str, float]:
    return {key: 0.6 for key in sim.robot_action_keys(TARGET)}


def _pose(sim: Any, robot: str) -> dict[str, float]:
    observation = sim.get_observation(robot_name=robot, skip_images=True)
    return {k: float(v) for k, v in observation.items() if not k.endswith(".vel")}


class TestTheLocalLoopDoor:
    """``teleoperate`` grades ``robot_name`` with its other loop knobs."""

    def test_an_unroutable_name_is_refused_and_connects_no_device(self, sim: Any) -> None:
        device = _leader(sim)
        sim.attach_teleop(device, name="leader")
        envelope = sim.teleoperate(robot_name=TYPO, hz=50, block=True, duration=0.6)

        assert envelope["status"] == "error"
        text = envelope["content"][0]["text"]
        assert TYPO in text
        assert f"Did you mean: {TARGET}" in text
        # Graded before the leader is energised, so there is nothing to roll back.
        assert device.connect_calls == 0
        assert device.is_connected is False

    def test_the_background_default_refuses_at_the_call_not_at_the_stop(self, sim: Any) -> None:
        """The pre-fix report: ``success`` now, an error only once stopped."""
        sim.attach_teleop(_leader(sim), name="leader")
        try:
            envelope = sim.teleoperate(robot_name=TYPO, hz=50)
            assert envelope["status"] == "error"
            # No session was started, so nothing is running to stop.
            status = next(b["json"] for b in sim.get_teleoperate_status()["content"] if "json" in b)
            assert status["running"] is False
            assert status["thread_alive"] is False
        finally:
            sim.stop_teleoperate()

    def test_the_named_robot_still_drives(self, sim: Any) -> None:
        """Control: the same call with the real name is unchanged."""
        sim.attach_teleop(_leader(sim), name="leader")
        envelope = sim.teleoperate(robot_name=TARGET, hz=50, block=True, duration=0.4)

        assert envelope["status"] == "success"
        telemetry = next(b["json"] for b in envelope["content"] if "json" in b)
        assert telemetry["frames"] > 0
        assert telemetry["errors"] == 0

    def test_no_name_in_a_multi_robot_world_is_still_accepted(self, sim: Any) -> None:
        """Control: ``None`` is resolved by the loop, not refused by the door."""
        sim.attach_teleop(_leader(sim), name="leader")
        envelope = sim.teleoperate(hz=50, block=True, duration=0.2)
        assert envelope["status"] == "success"


class TestTheMeshReceiveDoor:
    """``start_teleop_receive`` grades its third identifier ahead of the teardown."""

    def test_an_unroutable_name_leaves_the_live_stream_following(self, sim: Any) -> None:
        sim.mesh = _Mesh()
        assert sim.start_teleop_receive("leader-1", device_name="leader", robot_name=TARGET)["status"] == "success"
        key = "leader-1/leader"
        live = sim._input_receivers[key]

        envelope = sim.start_teleop_receive("leader-1", device_name="leader", robot_name=TYPO)

        assert envelope["status"] == "error"
        assert f"Did you mean: {TARGET}" in envelope["content"][0]["text"]
        # The stream that was following this leader is the one still registered,
        # and it still routes to the robot it was started for.
        assert sim._input_receivers[key] is live
        assert live.stats["running"] is True
        # ... and it still applies to the robot it was started for: one frame
        # through the live subscription is counted as delivered, not refused,
        # and it moves that arm.
        before = _pose(sim, TARGET)
        live._on_input(live.topic, {"t": time.time(), "seq": 0, "action": _command(sim)})
        assert live.stats["frames_received"] == 1
        assert live.stats["errors"] == 0
        assert max(abs(_pose(sim, TARGET)[j] - before[j]) for j in before) > 0
        live.stop()


class TestTheHostContractDefault:
    """A single-device host keeps the documented accept-and-ignore behaviour."""

    def test_any_name_is_routable_when_the_host_holds_one_device(self) -> None:
        host = _SingleDeviceHost()
        assert host._teleop_target_error("no-such-robot") is None

        host.attach_teleop(_Leader({"j.pos": 1.0}), name="leader")
        try:
            assert host.teleoperate(hz=200, robot_name="no-such-robot")["status"] == "success"
            deadline = time.time() + 2.0
            while time.time() < deadline and not host.sent:
                time.sleep(0.01)
            assert host.sent[-1] == "no-such-robot"
        finally:
            host.stop_teleoperate()
