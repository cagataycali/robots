"""Every Modbus TCP field the Robotiq codec cannot honour is refused, and named.

:mod:`strands_robots.drivers.robotiq.protocol` sits between a gripper tool call
and a socket, so it is the last place a bad field can be caught before it
becomes wire bytes -- and the first place a bad reply can be caught before it
becomes a position the gripper never reported. It has three refusal doors and
this pins all of them:

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

* the **header reader** refuses a header that cannot size a read. It is the
  earliest of the three and the only one that runs *before* the transport has
  the frame: a reply is read by the length its header declares, so an ungraded
  value there lets the peer decide how many bytes this client waits for, and
  every reason the parser would give arrives only once that wait is over. Its
  refusals are therefore graded on what they prevent as much as on what they
  say - no accepted header may ask for more than one PDU.

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
    MAX_MBAP_LENGTH,
    MAX_PDU_SIZE,
    MBAP_SIZE,
    MIN_MBAP_LENGTH,
    REGISTER_COUNT,
    FunctionCode,
    ProtocolError,
    aperture_mm_to_counts,
    mbap_body_size,
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


def _header(length: int, *, protocol_id: int = 0) -> bytes:
    """An MBAP header declaring ``length``, with nothing after it.

    The transport has only these seven bytes when it decides how many more to
    read, so a header is the whole input to that decision.
    """
    return struct.pack(">HHHB", 1, protocol_id, length, 9)


# A header that cannot size a read. Modbus TCP allows the length field a unit id
# plus at most a 253-byte PDU; a value outside that is not a short frame to be
# read and then rejected, it is a peer this client must not wait on.
HEADERS_THAT_CANNOT_SIZE_A_READ: tuple[tuple[str, bytes, str], ...] = (
    (
        "a header cut short carries no length to read by",
        _header(9)[:-1],
        rf"MBAP header must be {MBAP_SIZE} bytes, got {MBAP_SIZE - 1}",
    ),
    (
        "an http endpoint answering on the gripper's port",
        b"HTTP/1.",
        r"protocol id must be 0, got 21584 - this is not Modbus TCP",
    ),
    (
        "a length of zero leaves no room for the unit id",
        _header(0),
        rf"MBAP length must be in {MIN_MBAP_LENGTH}\.\.{MAX_MBAP_LENGTH}",
    ),
    (
        "a length of one leaves no room for a function code",
        _header(MIN_MBAP_LENGTH - 1),
        rf"MBAP length must be in {MIN_MBAP_LENGTH}\.\.{MAX_MBAP_LENGTH}",
    ),
    (
        "a length one past a whole PDU is more than Modbus can carry",
        _header(MAX_MBAP_LENGTH + 1),
        rf"MBAP length must be in {MIN_MBAP_LENGTH}\.\.{MAX_MBAP_LENGTH}",
    ),
    (
        "a length filling the field would hold the connection for 64KB",
        _header(0xFFFF),
        rf"MBAP length must be in {MIN_MBAP_LENGTH}\.\.{MAX_MBAP_LENGTH} .* got 65535",
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
    assert len(HEADERS_THAT_CANNOT_SIZE_A_READ) == 6


@pytest.mark.parametrize(
    ("header", "match"),
    [pytest.param(header, match, id=label) for label, header, match in HEADERS_THAT_CANNOT_SIZE_A_READ],
)
def test_a_header_that_cannot_size_a_read_is_refused_before_the_body(header: bytes, match: str) -> None:
    """The length is graded at the door that reads by it, not after the read.

    Every row here is a frame the parser would also refuse - but only once the
    body it declared had arrived, which for the last two rows means never. The
    refusal has to happen while the header is the only thing in hand.
    """
    with pytest.raises(ProtocolError, match=match):
        mbap_body_size(header)


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(MIN_MBAP_LENGTH, 1, id="the shortest legal reply is a bare function code"),
        pytest.param(MAX_MBAP_LENGTH, MAX_PDU_SIZE, id="the longest legal reply is one whole PDU"),
    ],
)
def test_the_edges_of_the_length_field_are_accepted_and_sized(length: int, expected: int) -> None:
    """Both ends of the domain are read, so the refusal is a domain and not a cap.

    A boundary asserted from one side only cannot tell a domain check from a
    limit that happens to sit somewhere below it.
    """
    assert mbap_body_size(_header(length)) == expected


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: read_input_registers_frame(1, 9, INPUT_BASE, REGISTER_COUNT), id="a read request"),
        pytest.param(lambda: write_registers_frame(1, 9, INPUT_BASE, (1, 2, 3)), id="a write request"),
        pytest.param(lambda: _reply(struct.pack(">BB", READ, 6) + b"\x00" * 6), id="a read reply"),
    ],
)
def test_the_size_reported_is_the_body_a_real_frame_carries(build: Callable[[], bytes]) -> None:
    """Graded against this module's own framing, so the unit-id offset is pinned.

    The declared length counts the unit id, which the header already carries, so
    the bytes still to read are one fewer. Asserting that against a frame built
    here means an off-by-one cannot pass by agreeing with itself.
    """
    frame = build()

    assert mbap_body_size(frame[:MBAP_SIZE]) == len(frame) - MBAP_SIZE


def test_no_header_can_ask_the_transport_for_more_than_one_pdu() -> None:
    """Exhaustive over the field: an accepted header sizes a bounded read.

    This is the property the refusals exist for, and it is worth stating over
    all 65536 values rather than at the edges: the read this sizes runs under
    the client's lock, and a socket timeout applies per ``recv``, not to the
    loop. So a length no larger than a PDU is what actually bounds the
    exchange - without it a peer that keeps dribbling bytes holds the
    connection for as long as it likes, whatever timeout the caller configured.
    """
    accepted = 0
    for length in range(0x10000):
        try:
            size = mbap_body_size(_header(length))
        except ProtocolError:
            continue
        accepted += 1
        assert 1 <= size <= MAX_PDU_SIZE, f"a header declaring {length} sized a read of {size} bytes"
    assert accepted == MAX_MBAP_LENGTH - MIN_MBAP_LENGTH + 1, "the accepted band is not the documented domain"
