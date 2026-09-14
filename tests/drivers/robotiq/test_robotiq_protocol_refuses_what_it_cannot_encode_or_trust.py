"""Every Modbus TCP field the Robotiq codec cannot honour is refused, and named.

:mod:`strands_robots.drivers.robotiq.protocol` sits between a gripper tool call
and a socket, so it is the last place a bad field can be caught before it
becomes wire bytes -- and the first place a bad reply can be caught before it
becomes a position the gripper never reported. It has two refusal halves and
this pins both:

* the **request builders** refuse a field they cannot encode, ahead of
  ``struct.pack``. Unguarded, six of these produce either a ``struct.error``
  naming a format character instead of the caller's parameter, a
  ``ZeroDivisionError``, or -- worse -- a frame that packs cleanly and means
  something else: ``transaction_id=True`` is an ``int`` to ``struct`` and would
  put transaction 1 on the wire, so a reply for an unrelated request 1 then
  passes the transaction check as this one's answer.
* the **reply parser** refuses a reply it cannot attribute or trust. Its checks
  read three independent sources of truth about one frame and are not
  redundant: the MBAP length grades the frame against *itself*, the byte count
  grades the payload against *the request*, and the carried length grades the
  payload against *its own declaration*. A reply can satisfy the first two and
  still carry half a register block, which decodes to a position the gripper
  never reported.

The already-pinned arms (a Modbus exception reply, a stale transaction id, a
non-Modbus protocol id, a byte count that disagrees with the request) live in
``test_robotiq_protocol_frames.py`` beside the byte-exact golden frames.
"""

from __future__ import annotations

import struct
from collections.abc import Callable

import pytest

from strands_robots.drivers.robotiq.protocol import (
    INPUT_BASE,
    MBAP_SIZE,
    REGISTER_COUNT,
    FunctionCode,
    ProtocolError,
    aperture_mm_to_counts,
    parse_response,
    read_input_registers_frame,
    read_registers_payload,
    write_registers_frame,
)

READ = FunctionCode.READ_INPUT_REGISTERS
WRITE = FunctionCode.WRITE_MULTIPLE_REGISTERS


def _reply(body: bytes, *, transaction_id: int = 1, unit_id: int = 9, length: int | None = None) -> bytes:
    """An MBAP-framed reply carrying ``body``.

    The declared length defaults to the one the body really needs (the unit id
    plus the PDU), so a frame built here passes the framing check and the test
    that uses it is about the check it names. ``length`` overrides that to build
    a frame whose header disagrees with its own contents.
    """
    declared = len(body) + 1 if length is None else length
    return struct.pack(">HHHB", transaction_id, 0, declared, unit_id) + body


# A field the caller supplied that cannot become wire bytes. Every row names the
# parameter in the message, because the alternative report is a struct.error
# quoting a format character the caller never wrote.
UNSENDABLE_REQUESTS: tuple[tuple[str, Callable[[], object], str], ...] = (
    (
        "stroke_mm of zero is not a stroke to clamp against",
        lambda: aperture_mm_to_counts(10.0, stroke_mm=0.0),
        r"stroke_mm must be positive, got 0\.0",
    ),
    (
        "a bool transaction id would silently become transaction 1",
        lambda: write_registers_frame(True, 9, INPUT_BASE, (1,)),  # type: ignore[arg-type]
        r"transaction_id must be an int, got True",
    ),
    (
        "a transaction id past 16 bits does not fit the MBAP field",
        lambda: write_registers_frame(0x10000, 9, INPUT_BASE, (1,)),
        r"transaction_id must be in 0\.\.65535, got 65536",
    ),
    (
        "a write of no registers writes nothing",
        lambda: write_registers_frame(1, 9, INPUT_BASE, ()),
        r"values must not be empty",
    ),
    (
        "a register value past 16 bits does not fit its word",
        lambda: write_registers_frame(1, 9, INPUT_BASE, (0x10000,)),
        r"register value must be an int in 0\.\.65535, got 65536",
    ),
    (
        "a read of no registers reads nothing",
        lambda: read_input_registers_frame(1, 9, INPUT_BASE, 0),
        r"count must be a positive int, got 0",
    ),
)


# A reply this codec cannot attribute to the request, or cannot trust the
# contents of. Each row is a different source of truth disagreeing.
UNTRUSTWORTHY_REPLIES: tuple[tuple[str, Callable[[], object], str], ...] = (
    (
        "a frame with no room for a function code carries no answer",
        lambda: parse_response(struct.pack(">HHHB", 1, 0, 3, 9), 1, READ),
        rf"response must be at least {MBAP_SIZE + 1} bytes, got {MBAP_SIZE}",
    ),
    (
        "an MBAP length that disagrees with the bytes that follow",
        lambda: parse_response(_reply(struct.pack(">BB", READ, 6), length=99), 1, READ),
        r"MBAP length says 99 bytes follow, got 3",
    ),
    (
        "a reply to a different function is not this request's answer",
        lambda: parse_response(_reply(struct.pack(">BHH", WRITE, INPUT_BASE, 3)), 1, READ),
        r"response function is 0x10, expected 0x04",
    ),
    (
        "a read reply with no byte count declares no register block",
        lambda: read_registers_payload(_reply(struct.pack(">B", READ)), 1, REGISTER_COUNT),
        r"read response carries no byte count",
    ),
    (
        "a read reply that declares the right block and carries less of it",
        lambda: read_registers_payload(_reply(struct.pack(">BB", READ, 6) + b"\x00\x01\x00"), 1, REGISTER_COUNT),
        r"read response declares 6 bytes but carries 3",
    ),
)


@pytest.mark.parametrize(
    ("build", "match"),
    [pytest.param(build, match, id=label) for label, build, match in UNSENDABLE_REQUESTS],
)
def test_a_field_that_cannot_go_on_the_wire_is_refused_by_name(build: Callable[[], object], match: str) -> None:
    """A frame builder answers the caller's parameter, not struct's format string."""
    with pytest.raises(ProtocolError, match=match):
        build()


@pytest.mark.parametrize(
    ("build", "match"),
    [pytest.param(build, match, id=label) for label, build, match in UNTRUSTWORTHY_REPLIES],
)
def test_a_reply_this_request_cannot_own_is_refused_by_name(build: Callable[[], object], match: str) -> None:
    """The parser refuses rather than handing back a PDU it cannot vouch for."""
    with pytest.raises(ProtocolError, match=match):
        build()


def test_the_refusal_tables_are_not_empty() -> None:
    """Guard against a table that silently becomes zero rows."""
    assert len(UNSENDABLE_REQUESTS) == 6
    assert len(UNTRUSTWORTHY_REPLIES) == 5
