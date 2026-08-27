# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bus broadcast address is not accepted as a single servo's ID.

:mod:`~strands_robots.tools.serial_tool` bounded ``motor_id`` at 254 and gave the
byte width as the reason ("the packet carries the ID in one byte, and 255 is the
frame header"). Two of the 256 values that byte holds do not address a servo at
all: ``0xFF`` is the frame header, and ``0xFE`` is the bus broadcast, which every
servo on the bus reads. Only the header was excluded.

Each of this tool's three motor-addressed actions writes to one servo and expects
its reply, so a broadcast is not a wider version of that request but a different
one:

===================  ==========================================================
``motor_id=254``     what reached the bus
===================  ==========================================================
``feetech_position`` every servo driven to the same angle, reported as
                     ``Feetech Motor 254 -> Position ...`` as though one moved
``feetech_velocity`` every servo run at the same speed, reported the same way
``feetech_ping``     every servo answering at once, colliding on the
                     half-duplex bus, then read back as one reply
===================  ==========================================================

The vendor draws the line in the same place. ``scservo_sdk`` declares
``BROADCAST_ID = 0xFE`` and returns ``COMM_NOT_AVAILABLE`` for ``scs_id >=
BROADCAST_ID`` in each of the three operations these actions perform - a ping, a
single register read and a single register write - and uses the broadcast only to
address ``SYNC_READ`` / ``SYNC_WRITE``, which this tool has no action for.
:class:`TestTheVendorSdkDrawsTheSameLine` pins that agreement whenever the SDK is
installed. ``strands_robots.drivers.feetech`` states the same ceiling as
``MAX_UNICAST_ID`` beside its own ``BROADCAST_ID``, and
:class:`TestTheToolAndTheCodecAgree` pins the two together.

Refusing the broadcast removes no capability: ``send`` writes the bytes it is
given and claims to address nothing, so a caller who wants a broadcast frame
still has one. :class:`TestBroadcastingIsStillReachable` grades that, so the
refusal cannot be widened into a ban on broadcasting.

The three sibling ``motor_id`` values below the broadcast are unchanged, and 255
is still refused for the reason it always was - this narrows the ceiling by one
value, the one that means "everybody".
"""

from __future__ import annotations

from typing import Any

import pytest
import serial

import strands_robots.tools.serial_tool as serial_mod

#: The bus broadcast address. Stated here rather than read off the module under
#: test so the premises and the controls grade the same wire semantics whether or
#: not the module has been fixed.
BUS_BROADCAST_ID = 0xFE

#: The frame header byte, which was already excluded.
FRAME_HEADER_BYTE = 0xFF

#: Largest ID that names one servo.
MAX_UNICAST_ID = 0xFD

#: IDs that address a single servo and must keep working.
UNICAST_IDS = (1, 2, 100, 252, MAX_UNICAST_ID)

#: The extra options each motor-addressed action needs to reach the port at all,
#: so a refusal in these tests is always the ``motor_id`` domain and never a
#: missing-argument branch.
_EXTRA_ARGS: dict[str, dict[str, Any]] = {
    "feetech_position": {"position": 100},
    "feetech_velocity": {"velocity": 100},
    "feetech_ping": {},
}


def _actions_taking_motor_id() -> tuple[str, ...]:
    """Every action the tool declares as consuming ``motor_id``.

    Derived from the tool's own option map rather than listed, so an action added
    later is held to this contract instead of inheriting an exemption by being
    absent from a tuple here.
    """
    return tuple(sorted(action for action, options in serial_mod._OPTIONS_BY_ACTION.items() if "motor_id" in options))


MOTOR_ADDRESSED_ACTIONS = _actions_taking_motor_id()

#: The actions that took ``motor_id`` when this contract was written. Stated so a
#: later edit that drops one is a failure rather than a quiet deselection.
KNOWN_MOTOR_ADDRESSED_ACTIONS = {"feetech_ping", "feetech_position", "feetech_velocity"}


class _FakeSerial:
    """Stand-in for ``serial.Serial`` recording the bytes that reach the wire."""

    def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.writes: list[bytes] = []
        self.closed = False
        self.in_waiting = 0

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    def read(self, n: int = 1) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[_FakeSerial]:
    """Record every port this tool opens; empty means the bus was never touched."""
    created: list[_FakeSerial] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> _FakeSerial:
        instance = _FakeSerial(port, baudrate, timeout)
        created.append(instance)
        return instance

    monkeypatch.setattr(serial, "Serial", _ctor)
    return created


def _call(action: str, motor_id: Any) -> dict[str, Any]:
    """Invoke a motor-addressed action with everything else already valid."""
    return serial_mod.serial_tool(action=action, port="/dev/fake0", motor_id=motor_id, **_EXTRA_ARGS[action])


def _text(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _written(opened: list[_FakeSerial]) -> list[bytes]:
    return [frame for port in opened for frame in port.writes]


class TestTheBroadcastIsNotAUnicastTarget:
    """The one value that means "every servo" is refused by every action."""

    @pytest.mark.parametrize("action", MOTOR_ADDRESSED_ACTIONS)
    def test_the_broadcast_is_refused(self, action: str, opened: list[_FakeSerial]) -> None:
        result = _call(action, BUS_BROADCAST_ID)
        assert result["status"] == "error"
        assert "motor_id" in _text(result)

    @pytest.mark.parametrize("action", MOTOR_ADDRESSED_ACTIONS)
    def test_no_frame_reaches_the_bus(self, action: str, opened: list[_FakeSerial]) -> None:
        _call(action, BUS_BROADCAST_ID)
        assert _written(opened) == []

    def test_the_documented_range_names_the_unicast_ceiling(self) -> None:
        """The range a caller reads must be the range the tool enforces."""
        doc = " ".join((serial_mod.serial_tool.__doc__ or "").split())
        assert f"an integer in [1, {MAX_UNICAST_ID}]" in doc

    @pytest.mark.parametrize("action", MOTOR_ADDRESSED_ACTIONS)
    def test_the_port_is_never_opened(self, action: str, opened: list[_FakeSerial]) -> None:
        """The module's first promise is that options are checked before the open."""
        _call(action, BUS_BROADCAST_ID)
        assert opened == []

    @pytest.mark.parametrize("action", MOTOR_ADDRESSED_ACTIONS)
    def test_the_refusal_names_the_broadcast(self, action: str, opened: list[_FakeSerial]) -> None:
        assert "broadcast" in _text(_call(action, BUS_BROADCAST_ID))

    @pytest.mark.parametrize("action", MOTOR_ADDRESSED_ACTIONS)
    def test_the_refusal_names_the_ceiling_that_replaces_it(self, action: str, opened: list[_FakeSerial]) -> None:
        assert f"at most {MAX_UNICAST_ID}" in _text(_call(action, BUS_BROADCAST_ID))


class TestEveryUnicastIdIsStillAccepted:
    """Narrowing the ceiling by one value must not cost the 253 below it."""

    @pytest.mark.parametrize("motor_id", UNICAST_IDS)
    @pytest.mark.parametrize("action", MOTOR_ADDRESSED_ACTIONS)
    def test_a_single_servo_is_addressed(self, action: str, motor_id: int, opened: list[_FakeSerial]) -> None:
        assert "motor_id" not in _text(_call(action, motor_id))

    @pytest.mark.parametrize("motor_id", UNICAST_IDS)
    @pytest.mark.parametrize("action", MOTOR_ADDRESSED_ACTIONS)
    def test_the_frame_carries_that_exact_id(self, action: str, motor_id: int, opened: list[_FakeSerial]) -> None:
        _call(action, motor_id)
        frames = _written(opened)
        assert frames, "an accepted id must reach the bus"
        assert frames[0][2] == motor_id


class TestTheFrameHeaderIsStillRefused:
    """The value the old ceiling did exclude is excluded for the same reason."""

    @pytest.mark.parametrize("action", MOTOR_ADDRESSED_ACTIONS)
    def test_the_header_byte_is_refused(self, action: str, opened: list[_FakeSerial]) -> None:
        result = _call(action, FRAME_HEADER_BYTE)
        assert result["status"] == "error"
        assert "motor_id" in _text(result)

    @pytest.mark.parametrize("action", MOTOR_ADDRESSED_ACTIONS)
    def test_the_refusal_still_names_the_frame_header(self, action: str, opened: list[_FakeSerial]) -> None:
        assert "frame header" in _text(_call(action, FRAME_HEADER_BYTE))


class TestBroadcastingIsStillReachable:
    """``send`` writes what it is given, so no capability is removed.

    This holds before and after the change. It is what stops the refusal above
    being widened into a ban on addressing the bus at all.
    """

    def test_a_hand_built_broadcast_frame_reaches_the_bus(self, opened: list[_FakeSerial]) -> None:
        frame = "FF FF FE 05 03 2A 64 00 6B"
        serial_mod.serial_tool(action="send", port="/dev/fake0", hex_data=frame)
        assert _written(opened) == [bytes.fromhex(frame.replace(" ", ""))]

    def test_send_does_not_read_motor_id_at_all(self) -> None:
        assert "motor_id" not in serial_mod._OPTIONS_BY_ACTION["send"]


class TestTheToolAndTheCodecAgree:
    """The tool and the package's Feetech codec state one ceiling, not two.

    The value is restated in the tool rather than imported: importing
    ``strands_robots.drivers.feetech`` executes ``drivers/__init__``, which
    registers every driver and pulls in numpy. This grades the agreement instead,
    which a test can afford and the lazily-loaded tool cannot.
    """

    def test_the_ceiling_is_the_codecs_max_unicast_id(self) -> None:
        from strands_robots.drivers.feetech import MAX_UNICAST_ID as codec_ceiling

        _, ceiling, _ = serial_mod._REGISTER_FIELDS["motor_id"]
        assert ceiling == codec_ceiling

    def test_the_excluded_value_is_the_codecs_broadcast_id(self) -> None:
        from strands_robots.drivers.feetech import BROADCAST_ID as codec_broadcast

        _, ceiling, _ = serial_mod._REGISTER_FIELDS["motor_id"]
        assert ceiling + 1 == codec_broadcast


class TestThisSuiteStatesTheWireSemanticsIndependently:
    """The constants above are the vendor's, so the controls grade real semantics.

    These hold before and after the change. They are what make the rest of this
    file an independent oracle rather than a restatement of the module under test.
    """

    def test_the_two_addresses_match_the_packages_codec(self) -> None:
        from strands_robots.drivers.feetech import BROADCAST_ID
        from strands_robots.drivers.feetech import MAX_UNICAST_ID as codec_ceiling

        assert (MAX_UNICAST_ID, BUS_BROADCAST_ID) == (codec_ceiling, BROADCAST_ID)

    def test_the_broadcast_sits_one_above_the_unicast_ceiling(self) -> None:
        assert BUS_BROADCAST_ID == MAX_UNICAST_ID + 1
        assert FRAME_HEADER_BYTE == BUS_BROADCAST_ID + 1


class TestTheVendorSdkDrawsTheSameLine:
    """The refused value is the vendor's broadcast, not this suite's opinion."""

    def test_the_sdk_declares_the_same_broadcast_address(self) -> None:
        scservo_def = pytest.importorskip("scservo_sdk.scservo_def")
        assert scservo_def.BROADCAST_ID == BUS_BROADCAST_ID

    def test_the_sdk_refuses_the_broadcast_for_reply_expecting_operations(self) -> None:
        pytest.importorskip("scservo_sdk")
        import inspect

        from scservo_sdk import protocol_packet_handler

        source = inspect.getsource(protocol_packet_handler)
        # ping, a single read and a single write each bail out before addressing
        # the frame when the caller named an id at or above the broadcast.
        assert source.count("if scs_id >= BROADCAST_ID:") >= 3
        assert "COMM_NOT_AVAILABLE" in source


class TestTheContractCoversEveryMotorAddressedAction:
    """The swept set is derived, and it is not empty."""

    def test_every_action_taking_motor_id_is_swept(self) -> None:
        assert set(MOTOR_ADDRESSED_ACTIONS) == set(_actions_taking_motor_id())

    def test_every_known_motor_addressed_action_is_still_swept(self) -> None:
        """Named here, so an action dropping ``motor_id`` fails instead of vanishing.

        The derived sweep above cannot see that: it reads the same option map the
        cells do, so removing an entry silently deselects its cases and the sweep
        still agrees with itself.
        """
        assert KNOWN_MOTOR_ADDRESSED_ACTIONS <= set(MOTOR_ADDRESSED_ACTIONS)

    def test_the_swept_set_is_not_empty(self) -> None:
        assert MOTOR_ADDRESSED_ACTIONS

    def test_every_swept_action_can_reach_the_port(self) -> None:
        assert set(MOTOR_ADDRESSED_ACTIONS) <= set(_EXTRA_ARGS), (
            "a new motor-addressed action needs its required options listed in "
            "_EXTRA_ARGS, or its refusal here would be a missing-argument branch"
        )
