"""``G1Driver.send_action``: writes ``LowCmd_`` on ``rt/lowcmd``.

The write half of issue #361: given a connected driver in a permissive FSM,
``send_action`` populates a ``LowCmd_`` message from the caller's ``action``
mapping and publishes it via the shared :class:`DDSPublisher`. The
subscribers set is exercised elsewhere; this suite pins:

* the SDK is not imported at driver import time (regression against #361),
* an unsafe FSM refuses before any wire touch,
* a shape error in ``action`` refuses before any wire touch,
* a well-shaped ``action`` reaches ``publisher.publish`` with the exact
  ``(topic, class, message)`` the caller intended,
* the exact per-motor values from ``action`` land on ``motor_cmd[joint]``.

No real ``unitree_sdk2py`` is loaded; a fake IDL and a fake ``ChannelPublisher``
stand in. The suite mirrors the fake-SDK shape in ``tests/tools/g1/test_dds_publisher.py``
so any future SDK convention change fails there and here together.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from strands_robots.drivers.g1 import (
    _LOWCMD_CLS_PATH,
    _TOPIC_LOWCMD,
    G1Driver,
)
from strands_robots.tools.g1 import HANDSHAKE_FSMS, reset_dds_state
from strands_robots.tools.g1._dds_engine import DDSPublisher

# =========================================================================
# Fixtures - a fake LowCmd_ IDL and a fake publisher the driver reaches.  #
# =========================================================================


class _FakeMotorCmd:
    """Records every field a caller set. One instance per motor slot.

    The fields are set via ``setattr`` from :meth:`G1Driver._build_lowcmd`, so
    they are *annotated* here but deliberately left unassigned. A bare
    annotation tells the type checker which fields the real IDL slot carries
    without creating them at runtime, which is what keeps the
    ``not hasattr(entry, "kp")`` assertions below honest: a field the driver
    never wrote is genuinely absent, exactly as an untouched IDL slot is.
    """

    q: float
    dq: float
    tau: float
    kp: float
    kd: float


class _FakeLowCmd:
    """Fake ``LowCmd_`` IDL: a ``motor_cmd`` list of :class:`_FakeMotorCmd`.

    The real IDL fixes the array length at construction; this mirrors that
    by pre-allocating :data:`_MOTOR_CMD_LEN` slots on ``__init__``. The
    number is arbitrary - the tests never assume the real firmware bound.
    """

    _MOTOR_CMD_LEN = 40

    def __init__(self) -> None:
        self.motor_cmd = [_FakeMotorCmd() for _ in range(self._MOTOR_CMD_LEN)]


class _RecordingPublisher(DDSPublisher):
    """Records every ``publish`` call the driver makes.

    Replaces :class:`~strands_robots.tools.g1._dds_engine.DDSPublisher` on
    the driver after ``connect_eagerly``, so the test does not need to
    mock the full DDS init lane.

    Subclasses the real engine rather than duck-typing it so that a future
    signature change to ``start``/``publish``/``close`` fails this suite at
    type-check time instead of letting the fake drift away from the object the
    driver actually calls.
    """

    def __init__(self) -> None:
        # The real __init__ only records the interface and builds a lock - no
        # DDS work happens until start(), which this class overrides away.
        super().__init__("lo")
        self.calls: list[tuple[str, type, Any]] = []
        self.next_error: str | None = None
        self.closed = False

    def start(self) -> str | None:
        return None

    def publish(self, topic: str, message_class: type, message: Any) -> str | None:
        self.calls.append((topic, message_class, message))
        return self.next_error

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_lowcmd_module(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install a fake ``unitree_sdk2py.idl.unitree_hg.msg.dds_`` for the test.

    The driver's ``_resolve_lowcmd_class`` does
    ``importlib.import_module(module_path)`` and reads ``LowCmd_`` off it,
    so this fixture creates that path and populates the attribute with
    :class:`_FakeLowCmd`.
    """
    module_path, class_name = _LOWCMD_CLS_PATH
    parts = module_path.split(".")
    # Build parent modules first (so a fresh import walks them).
    for i in range(1, len(parts) + 1):
        pkg_name = ".".join(parts[:i])
        if pkg_name not in sys.modules:
            monkeypatch.setitem(sys.modules, pkg_name, types.ModuleType(pkg_name))
    module = sys.modules[module_path]
    monkeypatch.setattr(module, class_name, _FakeLowCmd, raising=False)
    return _FakeLowCmd


@pytest.fixture
def connected_driver() -> Any:
    """A driver hoisted past its gates without touching real DDS.

    Sets ``_connected``, a walkable ``_fsm_id`` from :data:`HANDSHAKE_FSMS`,
    and a fresh :class:`_RecordingPublisher`. Cleared on teardown.
    """
    driver = G1Driver(tool_name="g1", port="192.168.1.172")
    driver._connected = True
    driver._fsm_id = next(iter(HANDSHAKE_FSMS))
    driver._publisher = _RecordingPublisher()
    yield driver
    reset_dds_state()


def _recorder(driver: G1Driver) -> _RecordingPublisher:
    """The fake publisher :func:`connected_driver` installed, precisely typed.

    ``G1Driver._publisher`` is declared ``DDSPublisher | None``, so reading the
    recording-only surface (``calls``, ``next_error``) straight off it is not
    type-safe. Narrowing in one place keeps every assertion below free of casts
    and turns the fixture's contract - that the driver is holding the fake and
    not a real publisher - into an assertion rather than an assumption.
    """
    publisher = driver._publisher
    assert isinstance(publisher, _RecordingPublisher)
    return publisher


# =========================================================================
# The class-level invariant that #361 buys: no SDK at import time.        #
# =========================================================================


def test_g1_driver_module_does_not_import_the_sdk() -> None:
    """Importing the driver must not touch ``unitree_sdk2py``.

    The whole point of the lazy IDL resolve is that a headless CI runner
    and Thor both import :class:`G1Driver` without the SDK installed.
    """
    import strands_robots.drivers.g1 as g1_mod

    ns = vars(g1_mod)
    # Attribute the SDK's namespace would leave if a module-level import ran.
    assert "unitree_sdk2py" not in ns
    assert "LowCmd_" not in ns
    assert "ChannelPublisher" not in ns


# =========================================================================
# Gates run first, before any wire touch.                                 #
# =========================================================================


def test_send_action_refuses_when_disconnected() -> None:
    """A driver that never connected refuses before touching the publisher."""
    driver = G1Driver(tool_name="g1", port="192.168.1.172")
    # No _connected, no _fsm_id, no _publisher.
    result = driver.send_action({"joints": [0], "q": [0.0]})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "not connected" in text


def test_send_action_refuses_when_fsm_is_wrong(connected_driver: G1Driver) -> None:
    """FSM outside :data:`HANDSHAKE_FSMS` refuses arm writes."""
    connected_driver._fsm_id = 200  # not in {500, 501, 801}
    result = connected_driver.send_action({"joints": [0], "q": [0.0]})
    assert result["status"] == "error"
    assert "arm writes" in result["content"][0]["text"]
    # And no publish happened.
    assert _recorder(connected_driver).calls == []


def test_send_action_refuses_under_battery_floor(connected_driver: G1Driver) -> None:
    """Battery under the floor refuses even when FSM is permissive."""
    connected_driver._battery = {"pct": 5.0}
    result = connected_driver.send_action({"joints": [0], "q": [0.0]})
    assert result["status"] == "error"
    assert "battery" in result["content"][0]["text"]
    assert _recorder(connected_driver).calls == []


# =========================================================================
# Shape errors refuse before the wire.                                    #
# =========================================================================


def test_send_action_refuses_a_bad_joints_type(
    connected_driver: G1Driver,
    fake_lowcmd_module: type,
) -> None:
    """``joints`` not coercible to ints refuses; no publish happens."""
    result = connected_driver.send_action({"joints": ["nope"]})
    assert result["status"] == "error"
    assert "joints" in result["content"][0]["text"]
    assert _recorder(connected_driver).calls == []


def test_send_action_refuses_a_length_mismatch(
    connected_driver: G1Driver,
    fake_lowcmd_module: type,
) -> None:
    """``q`` shorter or longer than ``joints`` refuses; no partial write."""
    result = connected_driver.send_action({"joints": [0, 1], "q": [0.5]})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "'q'" in text and "joints" in text
    assert _recorder(connected_driver).calls == []


def test_send_action_refuses_an_out_of_range_joint(
    connected_driver: G1Driver,
    fake_lowcmd_module: type,
) -> None:
    """A joint index past ``motor_cmd`` refuses; no partial write."""
    beyond = _FakeLowCmd._MOTOR_CMD_LEN + 5
    result = connected_driver.send_action({"joints": [beyond], "q": [0.0]})
    assert result["status"] == "error"
    assert "out of range" in result["content"][0]["text"]
    assert _recorder(connected_driver).calls == []


# =========================================================================
# The wire capture - a well-shaped write reaches publish, byte-identical. #
# =========================================================================


def test_send_action_writes_lowcmd_on_the_right_topic(
    connected_driver: G1Driver,
    fake_lowcmd_module: type,
) -> None:
    """The exact ``(topic, class, message)`` reaches the publisher.

    Publisher is mocked so this is a pure wire-capture: what the driver
    hands the DDS layer, byte-for-byte, is what this test reads back.
    """
    action = {
        "joints": [0, 5, 12],
        "q": [0.1, 0.2, 0.3],
        "kp": [50.0, 50.0, 50.0],
        "kd": [1.0, 1.0, 1.0],
    }
    result = connected_driver.send_action(action)
    assert result["status"] == "success"
    text = result["content"][0]["text"]
    assert "rt/lowcmd" in text
    assert "3 joint" in text
    calls = _recorder(connected_driver).calls
    assert len(calls) == 1
    topic, cls, message = calls[0]
    assert topic == _TOPIC_LOWCMD
    assert cls is fake_lowcmd_module
    assert isinstance(message, _FakeLowCmd)
    # Verify the per-motor fields landed exactly where the caller asked.
    assert message.motor_cmd[0].q == 0.1
    assert message.motor_cmd[5].q == 0.2
    assert message.motor_cmd[12].q == 0.3
    assert message.motor_cmd[0].kp == 50.0
    assert message.motor_cmd[5].kd == 1.0


def test_send_action_partial_fields_leave_others_at_default(
    connected_driver: G1Driver,
    fake_lowcmd_module: type,
) -> None:
    """``action`` with only ``q`` populates ``q`` and leaves other fields alone.

    The IDL default (whatever it is) is preserved for a field the caller
    did not name. The fake motor slots start bare, so the assertion is
    "the field was not set" via ``hasattr``.
    """
    result = connected_driver.send_action({"joints": [3], "q": [0.7]})
    assert result["status"] == "success"
    _, _, message = _recorder(connected_driver).calls[0]
    entry = message.motor_cmd[3]
    assert entry.q == 0.7
    assert not hasattr(entry, "kp")  # never set - IDL default preserved
    assert not hasattr(entry, "kd")
    assert not hasattr(entry, "tau")
    assert not hasattr(entry, "dq")


def test_send_action_empty_action_is_a_valid_stop(
    connected_driver: G1Driver,
    fake_lowcmd_module: type,
) -> None:
    """An action with no ``joints`` publishes an IDL-default ``LowCmd_``.

    That is the shape a caller uses to send a "hold" - every motor stays
    at whatever the IDL default is (0.0 across the board on the real IDL).
    """
    result = connected_driver.send_action({})
    assert result["status"] == "success"
    assert "0 joint" in result["content"][0]["text"]
    assert len(_recorder(connected_driver).calls) == 1
    _, _, message = _recorder(connected_driver).calls[0]
    # No fields set on any motor slot.
    for entry in message.motor_cmd:
        assert not hasattr(entry, "q")


def test_send_action_surfaces_a_publisher_error(
    connected_driver: G1Driver,
    fake_lowcmd_module: type,
) -> None:
    """When the DDS publish itself fails, ``send_action`` refuses with the reason."""
    _recorder(connected_driver).next_error = "publish to 'rt/lowcmd' failed: DDS dead"
    result = connected_driver.send_action({"joints": [0], "q": [0.0]})
    assert result["status"] == "error"
    assert "DDS dead" in result["content"][0]["text"]


# =========================================================================
# The LowCmd_ class is cached across calls.                               #
# =========================================================================


def test_lowcmd_class_is_resolved_once(
    connected_driver: G1Driver,
    fake_lowcmd_module: type,
) -> None:
    """A control loop at 500Hz must not re-import the IDL every step.

    First call resolves and caches; the second reads the cache. This is
    the same discipline :class:`DDSPublisher` uses for its ``(topic, cls)``
    cache - one construction, many uses.
    """
    assert connected_driver._lowcmd_class is None
    connected_driver.send_action({"joints": [0], "q": [0.0]})
    cached = connected_driver._lowcmd_class
    assert cached is fake_lowcmd_module
    connected_driver.send_action({"joints": [1], "q": [0.0]})
    # Same object - not just equal, identical - so a caller reading the
    # attribute sees no re-import happened.
    assert connected_driver._lowcmd_class is cached
