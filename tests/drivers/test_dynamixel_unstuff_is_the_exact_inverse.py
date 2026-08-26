"""The Protocol 2.0 escape pair is the identity, and the vendor's is not.

``strands_robots.drivers.dynamixel.protocol`` states in its module docstring
that every function is verifiable against Robotis' ``dynamixel_sdk``
byte-for-byte. That holds for the whole module except :func:`_unstuff`, whose
SDK counterpart ``removeStuffing`` compacts the packet in place and therefore
reads look-back positions its own loop has already overwritten. On a payload
carrying two reserved runs it matches an ``FF FF`` that is not there and drops
a data byte.

The module's docstring used to name ``removeStuffing`` as the implementation
this function mirrors, which is an instruction to port a byte loss in. Nothing
graded the divergence: with the SDK's algorithm substituted for the shipped
one, ``tests/drivers/test_dynamixel_driver.py`` is ``68 passed``.

These cells fix the adjudicator rather than the parity. An escape/unescape pair
that is not the identity is broken whichever implementation is older, so the
round trip is what is asserted here - over every payload that can carry a
reserved run, not over a hand-picked list. The two cells that consult the SDK
skip where it is absent (it is not a declared dependency of this project); the
identity cells need nothing but the module itself, so they run everywhere.
"""

from __future__ import annotations

import itertools

import pytest

from strands_robots.drivers.dynamixel import protocol

#: Bytes that can form or break the reserved ``FF FF FD`` run. A payload drawn
#: from any wider alphabet reduces to one of these for the purpose of stuffing,
#: so exhausting this alphabet exhausts the behaviour.
_RESERVED_ALPHABET = (0xFF, 0xFD, 0x00, 0x01)

#: Payload lengths short enough to exhaust and long enough to carry two runs.
#: Six is the shortest length at which the vendor's look-back can read a byte
#: it has already overwritten, so a corpus capped below six cannot see it.
_LENGTHS = (4, 5, 6, 7)

#: The shortest payload the vendor implementation mishandles. Two reserved runs
#: sharing the ``FD`` that ends the first: the second run's escape is inserted
#: at a position whose look-back the in-place compaction has already moved.
_WITNESS = bytes.fromhex("fffffdfffdfd")


def _frame(payload: bytes) -> bytes:
    """Wrap ``payload`` in a Protocol 2.0 frame with an unstuffed ``LEN``.

    Args:
        payload: The ``INST`` through last-parameter bytes.

    Returns:
        ``HEADER RESERVED ID LEN_L LEN_H`` followed by ``payload``, with
        ``LEN`` holding the count the format requires (payload plus the two
        CRC bytes that are appended later).
    """
    declared = len(payload) + 2
    header = (0xFF, 0xFF, 0xFD, 0x00, 0x01, declared & 0xFF, (declared >> 8) & 0xFF)
    return bytes(header) + payload


def _round_trip(payload: bytes) -> bytes:
    """Stuff ``payload`` into a frame and unstuff it back out.

    Args:
        payload: The bytes to send through both halves of the escape pair.

    Returns:
        What :func:`~strands_robots.drivers.dynamixel.protocol._unstuff`
        recovers from the stuffed frame.
    """
    stuffed = protocol._stuff(_frame(payload))
    return protocol._unstuff(stuffed[protocol._INST_INDEX :])


def _every_payload() -> list[bytes]:
    """Return every payload over the reserved alphabet at the graded lengths."""
    return [
        bytes(combination)
        for length in _LENGTHS
        for combination in itertools.product(_RESERVED_ALPHABET, repeat=length)
    ]


class TestTheEscapePairIsTheIdentity:
    """The property that makes the codec usable, asserted over a full corpus.

    This is the pin. It passes on the shipped implementation and fails on any
    edit that adopts the vendor's look-back, which is the edit the old
    docstring invited.
    """

    def test_the_corpus_is_large_enough_to_contain_a_double_run(self) -> None:
        payloads = _every_payload()
        assert len(payloads) == 21760
        doubles = [p for p in payloads if p.count(b"\xff\xff\xfd") >= 2]
        assert doubles, "a corpus with no doubled reserved run cannot see the divergence"
        assert _WITNESS in payloads

    def test_every_payload_survives_the_escape_pair_unchanged(self) -> None:
        lossy = [p.hex() for p in _every_payload() if _round_trip(p) != p]
        assert lossy == []

    def test_the_shortest_mishandled_payload_survives(self) -> None:
        assert _round_trip(_WITNESS) == _WITNESS

    def test_a_payload_with_no_reserved_run_is_returned_byte_for_byte(self) -> None:
        plain = bytes(range(0x10, 0x20))
        assert protocol._unstuff(plain) == plain


class TestTheVendorImplementationIsTheOneThatDiverges:
    """Grade the divergence the docstring now claims, against the SDK itself.

    Without these cells the docstring's account of ``removeStuffing`` is prose
    nobody checks, and a reader has no way to tell a deliberate divergence
    from drift.
    """

    @pytest.fixture
    def handler(self) -> object:
        sdk = pytest.importorskip(
            "dynamixel_sdk.protocol2_packet_handler",
            reason="dynamixel_sdk is not a declared dependency; the identity cells cover the module",
        )
        return sdk.Protocol2PacketHandler()

    @staticmethod
    def _vendor_unstuff(handler: object, stuffed: bytes) -> bytes:
        """Run the SDK's ``removeStuffing`` over a stuffed frame."""
        packet = list(stuffed) + [0x00, 0x00]
        out = handler.removeStuffing(packet)  # type: ignore[attr-defined]
        declared = out[5] | (out[6] << 8)
        return bytes(out[protocol._INST_INDEX : protocol._INST_INDEX + declared - 2])

    def test_the_two_stuffers_agree_byte_for_byte(self, handler: object) -> None:
        """The control: the divergence is in the unstuffer alone."""
        disagreed = []
        for payload in _every_payload():
            ours = protocol._stuff(_frame(payload))
            theirs = bytes(handler.addStuffing(list(_frame(payload)) + [0x00, 0x00]))  # type: ignore[attr-defined]
            if ours != theirs[: len(ours)]:
                disagreed.append(payload.hex())
        assert disagreed == []

    def test_the_vendor_unstuffer_drops_a_byte_on_the_witness(self, handler: object) -> None:
        stuffed = protocol._stuff(_frame(_WITNESS))
        assert self._vendor_unstuff(handler, stuffed) == bytes.fromhex("fffffdfffd")

    def test_the_vendor_round_trip_is_not_the_identity(self, handler: object) -> None:
        """Measured through the SDK's own stuffer, so our code is not in the loop."""
        lossy = []
        for payload in _every_payload():
            theirs = bytes(handler.addStuffing(list(_frame(payload)) + [0x00, 0x00]))  # type: ignore[attr-defined]
            if self._vendor_unstuff(handler, theirs[: len(protocol._stuff(_frame(payload)))]) != payload:
                lossy.append(payload.hex())
        assert len(lossy) == 12
        assert _WITNESS.hex() in lossy


class TestTheDocstringsNameTheDivergence:
    """A reader who follows the module's own instruction must not port the bug.

    These are the cells the change is for. The behaviour above was already
    correct; what was missing was any statement that the SDK is the wrong
    oracle for this one function, and any grading of that statement.
    """

    @staticmethod
    def _flat(text: str | None) -> str:
        """Collapse a docstring's wrapping so a phrase can be matched across lines."""
        return " ".join((text or "").split())

    def test_the_unstuff_docstring_does_not_present_the_vendor_as_its_model(self) -> None:
        doc = self._flat(protocol._unstuff.__doc__)
        assert "the mirror of Robotis' ``removeStuffing``" not in doc

    def test_the_unstuff_docstring_states_that_it_diverges_deliberately(self) -> None:
        doc = self._flat(protocol._unstuff.__doc__)
        assert "not** a port" in doc or "not* a port" in doc or "not a port" in doc
        assert "removeStuffing" in doc, "the divergence must name what it diverges from"

    def test_the_unstuff_docstring_warns_against_restoring_parity(self) -> None:
        doc = self._flat(protocol._unstuff.__doc__)
        assert "Do not" in doc and "removeStuffing" in doc

    def test_the_module_docstring_qualifies_its_byte_for_byte_claim(self) -> None:
        doc = self._flat(protocol.__doc__)
        assert "verifiable against Robotis' ``dynamixel_sdk`` byte-for -byte" in doc
        assert "exception" in doc, "an unqualified claim sends a reader to the wrong oracle"
