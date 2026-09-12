"""A Microduck action key no intent carries is refused, never part-sent.

robotd's surface is intent-level and every intent frame carries its *whole*
group: ``robot.move`` always carries ``vx``/``vy``/``vyaw``, ``robot.pose``
always carries ``z``/``roll``/``pitch``/``active``. So an unknown key beside a
known one is not ignored - it is filled in with the resting default and sent, and
the caller is told ``success``. ``{"vx": 0.15, "yaw": 0.6}`` walks straight past
the turn it asked for, and a 14-joint policy action reaches robotd as a head
frame built from four of its keys with the other ten dropped: exactly the
per-joint stream :meth:`MicroduckDriver.run_policy` refuses by name.

These tests pin the refusal over the spellings a caller actually reaches for -
this package's own state-read, sibling-driver and joint-name vocabularies - and
pin that an action whose every key is real still reaches the wire unchanged.
"""

from __future__ import annotations

import time

import pytest

from strands_robots.drivers.microduck import MicroduckDriver, action_to_wire
from strands_robots.policies.microduck import MICRODUCK_JOINT_NAMES
from tests.mocks.microduck_robotd import MockRobotd

#: ``(action, the keys that would have been dropped)``. Every unknown spelling
#: here is a name this package itself uses for the quantity being commanded.
DROPPED = [
    # get_status()'s pose block names the heading "yaw"; the twist key is "vyaw".
    ({"vx": 0.15, "yaw": 0.6}, ["yaw"]),
    # The Crazyflie driver's ACTION_KEYS spell a yaw rate "wz".
    ({"vx": 0.15, "wz": 0.6}, ["wz"]),
    # HARDWARE_JOINT_NAMES names the mouth joint "mouth"; the intent key is "open".
    ({"vx": 0.1, "mouth": 0.5}, ["mouth"]),
    # A plain typo inside a group that still parses.
    ({"z": 0.02, "rol": 0.1}, ["rol"]),
    ({"head_pitch": 0.3, "head_rol": 0.2}, ["head_rol"]),
]

#: The 14 joint targets a MicroduckPolicy produces. Four are head axes, so
#: without the gate this reaches robotd as a robot.head frame missing ten joints.
POLICY_ACTION = dict.fromkeys(MICRODUCK_JOINT_NAMES, 0.1)


class TestAnUnknownKeyBesideAKnownOne:
    """The action is refused, naming the key that would have been dropped."""

    @pytest.mark.parametrize("action, dropped", DROPPED, ids=[",".join(d) for _, d in DROPPED])
    def test_the_refusal_names_the_dropped_key(self, action: dict[str, object], dropped: list[str]) -> None:
        reason = action_to_wire(action)
        assert isinstance(reason, str), f"expected a refusal, got the frames {reason!r}"
        for key in dropped:
            assert repr(key) in reason, f"expected {key!r} named in {reason!r}"

    @pytest.mark.parametrize("action, dropped", DROPPED, ids=[",".join(d) for _, d in DROPPED])
    def test_the_refusal_names_the_vocabulary_to_use_instead(
        self, action: dict[str, object], dropped: list[str]
    ) -> None:
        reason = action_to_wire(action)
        assert isinstance(reason, str)
        for key in ("vx", "vyaw", "head_roll", "open", "skill"):
            assert key in reason, f"expected the accepted key {key!r} offered in {reason!r}"

    def test_a_policy_joint_action_is_refused_naming_the_joints_no_intent_carries(self) -> None:
        reason = action_to_wire(dict(POLICY_ACTION))
        assert isinstance(reason, str), f"expected a refusal, got the frames {reason!r}"
        assert "'left_knee'" in reason and "'right_ankle'" in reason


class TestTheRefusalsPartition:
    """Each refusal diagnoses one fault, and a whole action still flies."""

    def test_an_action_naming_no_intent_at_all_is_told_what_to_send(self) -> None:
        # Kept distinct from the unknown-key refusal: nothing parsed, so the
        # caller needs the vocabulary, not a list of their own keys.
        assert action_to_wire({"forward": 0.3}) == []

    def test_an_action_whose_every_key_is_real_reaches_the_wire(self) -> None:
        commands = action_to_wire(
            {
                "vx": 0.1,
                "vy": 0.0,
                "vyaw": 0.2,
                "z": 0.02,
                "roll": 0.0,
                "pitch": 0.0,
                "active": True,
                "open": 0.5,
                "skill": "kick_left",
            }
        )
        assert not isinstance(commands, str), commands
        assert [method for method, _params, _notify in commands] == [
            "robot.move",
            "robot.pose",
            "robot.mouth",
            "robot.do",
        ]


def test_nothing_reaches_a_real_robotd_socket_for_a_dropped_key() -> None:
    """Over a genuine socket: an error envelope, and no frame was written."""
    with MockRobotd() as server:
        driver = MicroduckDriver(tool_name="microduck", port=server.path, timeout=2.0)
        assert driver.connect_eagerly() is None
        try:
            result = driver.send_action({"vx": 0.15, "yaw": 0.6})
            assert result["status"] == "error", result
            assert "'yaw'" in result["content"][0]["text"]
            time.sleep(0.1)  # a written notification would have landed by now
            assert not [line for line in server.received if b"robot.move" in line], server.methods
        finally:
            driver.cleanup()


def test_an_action_naming_no_intent_is_refused_by_the_driver() -> None:
    """The no-known-key path still reports its own reason through send_action."""
    with MockRobotd() as server:
        driver = MicroduckDriver(tool_name="microduck", port=server.path, timeout=2.0)
        assert driver.connect_eagerly() is None
        try:
            result = driver.send_action({"forward": 0.3})
            assert result["status"] == "error"
            assert "nothing to send" in result["content"][0]["text"]
        finally:
            driver.cleanup()
