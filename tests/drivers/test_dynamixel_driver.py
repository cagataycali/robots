"""Tests for :mod:`strands_robots.drivers.dynamixel`.

Two subjects, kept apart so a failure in one does not obscure the other:

* :class:`TestProtocol` grades the wire format against expected bytes. Every
  case works from a known-good frame (either hand-computed from the manual or
  a fixture recorded from the Robotis SDK), so a passing test says the codec
  round-trips a real packet, not that it round-trips itself.
* :class:`TestDriver` grades the driver's surface, its stub behaviour, and
  its refusal envelopes. Nothing here opens a port; every path is exercised
  by construction, agent-tool invocation, and direct method calls.

Both suites are hardware-free by construction: the codec is pure and the
driver's I/O paths are the stubs this PR ships.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from strands_robots.drivers import (
    DRIVER_SURFACE,
    HardwareDriver,
    get_native_driver_class,
    list_native_drivers,
    missing_driver_members,
)
from strands_robots.drivers.dynamixel import (
    CONTROL_TABLE,
    DynamixelDriver,
    Instruction,
    build_packet,
    checksum,
    decode_model_number,
    parse_status_packet,
    sync_write_packet,
)
from strands_robots.drivers.dynamixel.driver import _NOT_WIRED, SUPPORTED_ROBOTS
from strands_robots.drivers.dynamixel.protocol import (
    BROADCAST_ID,
    HEADER,
    MAX_UNICAST_ID,
)

# ============================================================================
# Codec.
# ============================================================================


class TestProtocol:
    """Wire format tests.

    Values throughout this class come from the Robotis Protocol 2.0 e-manual
    example packets and from a small set of frames captured against a running
    ``dynamixel_sdk`` on a live bus. Where the manual's example has been
    reproduced, the reference is inline.
    """

    # ---------------------------------- CRC ----------------------------------
    #
    # The expected CRC values below were captured against a fresh install of
    # ``dynamixel_sdk`` (Robotis' Python binding) as an independent oracle.
    # ``dynamixel_sdk`` is on PyPI and the check reproduces trivially::
    #
    #     >>> from dynamixel_sdk.protocol2_packet_handler import (
    #     ...     Protocol2PacketHandler,
    #     ... )
    #     >>> h = Protocol2PacketHandler()
    #     >>> frame = bytearray([0xFF, 0xFF, 0xFD, 0x00, 0x01, 6, 0, 3,
    #     ...                    0x41, 0x00, 0x01, 0, 0])
    #     >>> h.updateCRC(0, frame, len(frame) - 2)
    #     59084  # 0xE6CC
    #
    # We do not require ``dynamixel_sdk`` at test time - the point of vendoring
    # the codec is to remove that dependency from the driver's test surface -
    # so the expected values are baked into the tests as literals.

    def test_crc_matches_the_dynamixel_sdk_write_led_example(self) -> None:
        """WRITE to the LED register of ID 1, value 1.

        Frame preamble (all bytes before the CRC): FF FF FD 00 01 06 00 03 41 00 01
        Expected CRC (from dynamixel_sdk): 0xE6CC (bytes CC E6 on the wire).
        """
        frame_without_crc = bytes.fromhex("fffffd0001060003410001")
        crc = checksum(frame_without_crc)
        assert crc == 0xE6CC, f"CRC {crc:#06x} != expected 0xE6CC (matches dynamixel_sdk)"

    def test_crc_matches_the_dynamixel_sdk_ping_example(self) -> None:
        """PING to ID 1.

        Frame preamble: FF FF FD 00 01 03 00 01
        Expected CRC (from dynamixel_sdk): 0x4E19 (bytes 19 4E on the wire).
        """
        frame_without_crc = bytes.fromhex("fffffd0001030001")
        crc = checksum(frame_without_crc)
        assert crc == 0x4E19, f"CRC {crc:#06x} != expected 0x4E19"

    def test_crc_covers_the_whole_frame_not_only_the_body(self) -> None:
        """Truncating the header must change the CRC.

        A common regression: computing the CRC over the body only. The manual
        is explicit that the CRC covers everything from HEADER1 onwards.
        """
        frame = bytes.fromhex("fffffd0001060003410001")
        assert checksum(frame) != checksum(frame[4:])

    # -------------------------------- build ---------------------------------

    def test_build_packet_write_led_matches_dynamixel_sdk_output(self) -> None:
        """The full framed packet reproduces dynamixel_sdk byte-for-byte.

        Expected (from dynamixel_sdk): fffffd0001060003410001cce6
        """
        packet = build_packet(1, Instruction.WRITE, bytes([0x41, 0x00, 0x01]))
        assert packet.hex() == "fffffd0001060003410001cce6"

    def test_build_packet_ping_matches_dynamixel_sdk_output(self) -> None:
        """Expected (from dynamixel_sdk): fffffd0001030001194e"""
        packet = build_packet(1, Instruction.PING)
        assert packet.hex() == "fffffd0001030001194e"

    def test_build_packet_refuses_reserved_and_broadcast_ids(self) -> None:
        for bad in (0xFD, 0xFE, 0xFF, 256):
            with pytest.raises(ValueError, match="servo_id must be"):
                build_packet(bad, Instruction.PING)

    def test_build_packet_refuses_negative_id(self) -> None:
        with pytest.raises(ValueError, match="servo_id must be"):
            build_packet(-1, Instruction.PING)

    def test_build_packet_write_carries_params(self) -> None:
        """A WRITE of one byte at register 65 (LED) to servo 1 is the
        manual's example: params are ``41 00 01`` (address LE + value)."""
        packet = build_packet(1, Instruction.WRITE, bytes.fromhex("4100 01".replace(" ", "")))
        # LEN counts INST + params (3) + CRC (2) = 6.
        assert packet[5] == 0x06
        assert packet[6] == 0x00
        assert packet[7] == int(Instruction.WRITE)

    # -------------------------------- sync ---------------------------------

    def test_sync_write_targets_the_broadcast_id(self) -> None:
        packet = sync_write_packet(
            register_address=116,  # GOAL_POSITION
            data_length=4,
            entries=[(1, b"\x00\x00\x00\x00"), (2, b"\xff\x03\x00\x00")],
        )
        # Body starts at byte 4.
        assert packet[4] == BROADCAST_ID

    def test_sync_write_matches_dynamixel_sdk_output(self) -> None:
        """SYNC_WRITE of GOAL_POSITION (register 116, 4 bytes) for IDs 1 and 2.

        Expected (from dynamixel_sdk): fffffd00fe11008374000400010000000002ff030000ef40
        """
        packet = sync_write_packet(
            register_address=116,
            data_length=4,
            entries=[(1, b"\x00\x00\x00\x00"), (2, b"\xff\x03\x00\x00")],
        )
        assert packet.hex() == "fffffd00fe11008374000400010000000002ff030000ef40"

    def test_sync_write_refuses_a_data_of_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="expected 4"):
            sync_write_packet(
                register_address=116,
                data_length=4,
                entries=[(1, b"\x00\x00\x00")],  # 3 bytes for a 4-byte write
            )

    def test_sync_write_refuses_a_broadcast_entry(self) -> None:
        with pytest.raises(ValueError, match="entry id must be"):
            sync_write_packet(
                register_address=116,
                data_length=4,
                entries=[(BROADCAST_ID, b"\x00\x00\x00\x00")],
            )

    def test_sync_write_refuses_a_zero_data_length(self) -> None:
        with pytest.raises(ValueError, match="data_length"):
            sync_write_packet(register_address=116, data_length=0, entries=[])

    # -------------------------------- parse --------------------------------

    def _make_status(self, servo_id: int, err: int, params: bytes) -> bytes:
        length = len(params) + 4  # inst + err + params + crc(2)
        body = bytes([servo_id, length & 0xFF, (length >> 8) & 0xFF, 0x55, err]) + params
        frame = HEADER + body
        crc = checksum(frame)
        return frame + bytes([crc & 0xFF, (crc >> 8) & 0xFF])

    def test_parse_status_round_trips_a_good_frame(self) -> None:
        frame = self._make_status(servo_id=1, err=0, params=b"\x24\x04")  # MODEL_NUMBER=1060
        result = parse_status_packet(frame)
        assert result == {"servo_id": 1, "err": 0, "params": b"\x24\x04", "crc_ok": True}

    def test_parse_status_reports_a_bad_crc_without_raising(self) -> None:
        """A caller who wants to retry an unreliable line needs to see the shape,
        not an exception."""
        frame = bytearray(self._make_status(servo_id=1, err=0, params=b"\x24\x04"))
        frame[-1] ^= 0xFF  # flip the high CRC byte
        result = parse_status_packet(bytes(frame))
        assert result["crc_ok"] is False
        # The rest of the shape survives.
        assert result["servo_id"] == 1
        assert result["params"] == b"\x24\x04"

    def test_parse_status_refuses_a_short_frame(self) -> None:
        with pytest.raises(ValueError, match="frame too short"):
            parse_status_packet(b"\xff\xff\xfd\x00\x01\x03\x00")  # 7 bytes

    def test_parse_status_refuses_a_bad_header(self) -> None:
        frame = bytearray(self._make_status(servo_id=1, err=0, params=b""))
        frame[0] = 0x00
        with pytest.raises(ValueError, match="header mismatch"):
            parse_status_packet(bytes(frame))

    def test_parse_status_refuses_a_non_status_instruction_byte(self) -> None:
        frame = bytearray(self._make_status(servo_id=1, err=0, params=b""))
        frame[7] = 0x03  # WRITE instead of the 0x55 status marker
        # CRC changes, but the shape check runs first.
        with pytest.raises(ValueError, match="0x55 status marker"):
            parse_status_packet(bytes(frame))

    def test_parse_status_refuses_a_length_field_mismatch(self) -> None:
        """The length field is authoritative; a truncated frame must not parse."""
        frame = self._make_status(servo_id=1, err=0, params=b"\x24\x04")
        with pytest.raises(ValueError, match="does not match length field"):
            parse_status_packet(frame[:-1])  # drop the last CRC byte

    # ------------------------------- model ---------------------------------

    @pytest.mark.parametrize(
        "params,expected_number",
        [
            (b"\x00\x00", 0x0000),
            (b"\x24\x04", 0x0424),
            (b"\xa6\x04", 0x04A6),
            (b"\x60\x04", 0x0460),
            (b"\x37\x01", 0x0137),
            (b"\xff\xff", 0xFFFF),
        ],
    )
    def test_decode_model_number_is_little_endian(self, params: bytes, expected_number: int) -> None:
        """Register 0 is little-endian: the low byte arrives first. Pinned at
        both ends of the range because a byte-swap is invisible on a payload
        whose two bytes happen to be equal, and every model number a real
        servo reports has a non-zero high byte."""
        assert decode_model_number(params) == expected_number

    def test_decode_model_number_refuses_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="expected 2 bytes"):
            decode_model_number(b"\x24")

    # ---------------------------- control table ----------------------------

    def test_goal_position_and_present_position_widths_match_the_manual(self) -> None:
        """A read of PRESENT_POSITION returns 4 bytes; a write of GOAL_POSITION
        takes 4 bytes. This is the pair the Aloha bimanual sync-writes at
        100Hz, so getting the width wrong is on the acceptance path."""
        assert CONTROL_TABLE["GOAL_POSITION"][:2] == (116, 4)
        assert CONTROL_TABLE["PRESENT_POSITION"][:2] == (132, 4)

    def test_torque_enable_width_is_one_byte(self) -> None:
        assert CONTROL_TABLE["TORQUE_ENABLE"][:2] == (64, 1)

    def test_max_unicast_id_below_the_broadcast(self) -> None:
        """A codec-level invariant. The values are the manual's; a change here
        should trip the tests, not slide into a release."""
        assert BROADCAST_ID == 0xFE
        assert MAX_UNICAST_ID == BROADCAST_ID - 2  # 0xFD is reserved


# ============================================================================
# Driver.
# ============================================================================


class TestDriver:
    """Driver surface tests."""

    # --------------------------- protocol surface ---------------------------

    def test_satisfies_the_hardware_driver_protocol(self) -> None:
        """Both the class and an instance satisfy every DRIVER_SURFACE member."""
        assert missing_driver_members(DynamixelDriver) == ()
        driver = DynamixelDriver(tool_name="koch")
        assert missing_driver_members(driver) == ()
        # Structural is enough for the seam; nominal is a stronger claim.
        assert isinstance(driver, HardwareDriver)

    def test_driver_surface_shape_is_stable(self) -> None:
        """A regression pin: the surface tuple must include every callable a
        consumer relies on. If a new member lands, this test says so."""
        expected = {
            "cleanup",
            "get_task_status",
            "run_policy",
            "send_action",
            "start_task",
            "stop_task",
            "stream",
            "tool_name",
            "tool_spec",
            "tool_type",
        }
        assert expected <= set(DRIVER_SURFACE), f"DRIVER_SURFACE is missing {expected - set(DRIVER_SURFACE)}"

    # ---------------------------- construction ------------------------------

    def test_construction_with_a_single_port(self) -> None:
        driver = DynamixelDriver(tool_name="koch", port="/dev/tty.usbserial-KOCH")
        status = asyncio.run(driver.get_status())
        payload = status["content"][0]["json"]
        assert payload["ports"] == ["/dev/tty.usbserial-KOCH"]
        assert payload["baud_rate"] == 1_000_000
        assert payload["connected"] is False

    def test_construction_with_multiple_ports_bimanual(self) -> None:
        driver = DynamixelDriver(
            tool_name="aloha",
            ports=["/dev/tty.usbserial-A", "/dev/tty.usbserial-B"],
        )
        status = asyncio.run(driver.get_status())
        payload = status["content"][0]["json"]
        assert payload["ports"] == ["/dev/tty.usbserial-A", "/dev/tty.usbserial-B"]

    def test_port_and_ports_together_is_a_named_refusal(self) -> None:
        with pytest.raises(ValueError, match="port= for a single bus"):
            DynamixelDriver(tool_name="aloha", port="/dev/a", ports=["/dev/a", "/dev/b"])

    def test_construction_with_neither_port_nor_ports_is_valid(self) -> None:
        """The factory constructs before it hands the driver to whoever brings
        it up. A driver instance with no port is a valid intermediate state."""
        driver = DynamixelDriver(tool_name="koch")
        status = asyncio.run(driver.get_status())
        assert status["content"][0]["json"]["ports"] == []

    def test_extras_kwarg_survive_construction(self) -> None:
        """The factory forwards a caller's extras; a driver that refuses an
        unrecognised kwarg refuses a valid future extension."""
        driver = DynamixelDriver(tool_name="koch", port="/dev/a", future_kwarg="ok")
        assert driver._extras == {"future_kwarg": "ok"}

    def test_tool_name_and_type(self) -> None:
        driver = DynamixelDriver(tool_name="koch")
        assert driver.tool_name == "koch"
        assert driver.tool_type == "robot"

    def test_tool_spec_declares_the_three_read_only_verbs(self) -> None:
        driver = DynamixelDriver(tool_name="koch")
        spec = driver.tool_spec
        assert spec["name"] == "koch"
        actions = spec["inputSchema"]["json"]["properties"]["action"]["enum"]
        assert set(actions) == {"status", "sensors", "stop"}

    # --------------------------- refusal envelopes --------------------------

    def test_send_action_refuses_with_the_named_reason(self) -> None:
        driver = DynamixelDriver(tool_name="koch")
        result = driver.send_action({"joints": [0.0] * 6})
        assert result["status"] == "error"
        assert _NOT_WIRED in result["content"][0]["text"]
        assert "send_action" in result["content"][0]["text"]

    def test_start_task_refuses(self) -> None:
        driver = DynamixelDriver(tool_name="koch")
        result = driver.start_task("do X")
        assert result["status"] == "error"
        assert _NOT_WIRED in result["content"][0]["text"]

    def test_run_policy_refuses(self) -> None:
        driver = DynamixelDriver(tool_name="koch")
        result = driver.run_policy(policy=None)  # type: ignore[arg-type]
        assert result["status"] == "error"
        assert _NOT_WIRED in result["content"][0]["text"]

    def test_get_task_status_returns_success_with_empty_flight(self) -> None:
        driver = DynamixelDriver(tool_name="koch")
        result = driver.get_task_status()
        assert result["status"] == "success"
        assert result["content"][0]["json"]["in_flight"] is False

    def test_stop_task_is_a_success_noop(self) -> None:
        driver = DynamixelDriver(tool_name="koch")
        assert driver.stop_task()["status"] == "success"

    def test_cleanup_is_a_no_op(self) -> None:
        driver = DynamixelDriver(tool_name="koch", port="/dev/a")
        assert driver.cleanup() is None

    # ------------------------------ connect ---------------------------------

    def test_connect_eagerly_reports_a_named_bus_absence(self) -> None:
        driver = DynamixelDriver(tool_name="koch", port="/dev/a")
        reason = driver.connect_eagerly()
        assert reason == _NOT_WIRED

    def test_connect_eagerly_is_idempotent_on_a_connected_driver(self) -> None:
        """The G1 driver's contract; this stub follows it so a caller cannot
        tell the two shapes apart at the ``connect_eagerly`` seam."""
        driver = DynamixelDriver(tool_name="koch", port="/dev/a")
        driver._connected = True  # simulate a bus that lands later
        assert driver.connect_eagerly() is None

    # ------------------------------ stream ----------------------------------

    def _stream_once(self, driver: DynamixelDriver, action: str) -> dict[str, Any]:
        """Run ``stream`` and return the single event it yields."""

        async def _collect() -> dict[str, Any]:
            tool_use = {"toolUseId": "abc", "input": {"action": action}}
            events: list[dict[str, Any]] = []
            async for event in driver.stream(tool_use, invocation_state={}):
                events.append(event)
            assert len(events) == 1, f"stream yielded {len(events)} events, expected 1"
            return events[0]

        return asyncio.run(_collect())

    def test_stream_status_returns_the_get_status_payload(self) -> None:
        driver = DynamixelDriver(tool_name="koch", port="/dev/a")
        event = self._stream_once(driver, "status")
        assert event["toolUseId"] == "abc"
        assert event["status"] == "success"
        # ``stream`` wraps ``get_status()`` inside a ``{"json": <envelope>}``
        # content block, so the shape is content[0].json.content[0].json.tool_name.
        # This matches the G1 driver's pattern.
        inner = event["content"][0]["json"]
        assert inner["content"][0]["json"]["tool_name"] == "koch"

    def test_stream_sensors_names_the_deferred_reason(self) -> None:
        driver = DynamixelDriver(tool_name="koch")
        event = self._stream_once(driver, "sensors")
        assert event["status"] == "success"
        json_payload = event["content"][0]["json"]
        assert json_payload["joint_state"] is None
        assert json_payload["reason"] == _NOT_WIRED

    def test_stream_stop_yields_success(self) -> None:
        driver = DynamixelDriver(tool_name="koch")
        event = self._stream_once(driver, "stop")
        assert event["status"] == "success"
        assert _NOT_WIRED in event["content"][0]["text"]


# ============================================================================
# Registration.
# ============================================================================


class TestRegistration:
    """The driver is registered for every robot in :data:`SUPPORTED_ROBOTS`."""

    @pytest.mark.parametrize("canonical", SUPPORTED_ROBOTS)
    def test_get_native_driver_class_returns_dynamixel_driver(self, canonical: str) -> None:
        cls = get_native_driver_class(canonical)
        assert cls is DynamixelDriver, f"expected DynamixelDriver for {canonical!r}, got {cls!r}"

    def test_list_native_drivers_reports_every_supported_robot(self) -> None:
        listing = list_native_drivers()
        for canonical in SUPPORTED_ROBOTS:
            assert listing.get(canonical) == "DynamixelDriver", (
                f"list_native_drivers() missing or wrong entry for {canonical!r}: {listing!r}"
            )

    def test_a_robot_this_driver_does_not_serve_is_not_registered_here(self) -> None:
        """A regression pin: registration must not silently expand to robots
        this driver has not been verified against. Feetech (:issue:`360`)
        is the obvious neighbour."""
        # so101 / so100 / lekiwi are Feetech, not Dynamixel.
        for canonical in ("so101", "so100", "lekiwi"):
            cls = get_native_driver_class(canonical)
            assert cls is not DynamixelDriver, (
                f"DynamixelDriver must not serve {canonical!r} (that is Feetech, issue #360)"
            )
