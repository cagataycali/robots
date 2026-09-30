"""Prerequisites an operator must satisfy to recover a fleet from an e-stop.

``Mesh.emergency_stop`` latches a lockout on every peer that receives it, and
nothing clears it on a timer - an e-stop that expired by itself would not be an
e-stop. The only way back is an explicit ``resume``, so every precondition that
resume depends on is a precondition for recovering the fleet at all.

Two of those preconditions are invisible until the fleet is already locked out:

1. ``STRANDS_MESH_OVERRIDE_CODE`` must be set on every peer. The mesh already
   logs a WARNING at startup when it is unset.
2. Fleet clocks must agree. A resume envelope carries the issuer's wall clock and
   a receiver refuses one that is stale (older than
   ``STRANDS_MESH_RESUME_FRESHNESS_S``) or future-dated (more than
   ``STRANDS_MESH_RESUME_FORWARD_SKEW_S`` ahead). The forward bound is the tight
   one and the asymmetry is the trap: a receiver whose clock is a few seconds
   *behind* the operator reads a correct, correctly-signed resume as future-dated
   and refuses it, and every retry fails identically. Nothing in the process
   recovers from that - the fleet stays stopped until the clock is corrected or
   the bound is widened on every peer.

The clock precondition has no startup warning, so documentation is the only place
an operator can learn it before it matters. These tests pin the behaviour and pin
that the knobs it depends on are documented, so the numbers in the docs cannot
drift away from the numbers the receiver enforces.

Two docs surfaces carry the claims. ``docs/reference/configuration.md`` is the
generated env-var table (its ``{{env_vars}}`` token is expanded here through
``docs/hooks/env_vars.py``, the way mkdocs does it), and it must list every knob.
``docs/learn/mesh/safety-and-estop.md`` is the page that shows ``emergency_stop()``
and the resume envelope; it is where the operator-facing prose lives, so the
sentence on it that names each skew bound is where the clock direction is read.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import strands_robots
from strands_robots.mesh import core
from tests._docs_hooks import docs_hook

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_CONFIGURATION = _REPO_ROOT / "docs" / "reference" / "configuration.md"
_SAFETY_PAGE = _REPO_ROOT / "docs" / "learn" / "mesh" / "safety-and-estop.md"

#: Long enough to be a realistic operator secret rather than a crackable PIN.
_CODE = "operator-code-1234567890abcdef"


def _mesh(peer_id: str) -> Any:
    """A Mesh wired for the safety handlers without joining a real session."""
    m = core.Mesh.__new__(core.Mesh)
    core.Mesh.__init__(m, MagicMock(), peer_id)
    m.publish_safety_event = lambda **kw: None  # type: ignore[method-assign]
    return m


def _mint_resume(*, issuer_clock_offset_s: float = 0.0) -> dict[str, Any]:
    """Return the resume envelope the real issuer publishes.

    ``issuer_clock_offset_s`` models an operator whose wall clock runs ahead of
    the receiver's, which is the same relative skew as a receiver running behind.
    The offset is applied only while the issuer mints the envelope, so the
    receiver under test always runs on the real clock.
    """
    operator = _mesh("operator-1")
    captured: dict[str, Any] = {}
    operator._publish_safety_envelope = lambda key, env: captured.update(env)
    operator._estop_lockout.set()
    operator._last_estop_ts = time.time() - 3.0
    operator._last_estop_mono = time.monotonic() - 3.0
    real_time = time.time
    with patch.object(core.time, "time", lambda: real_time() + issuer_clock_offset_s):
        result = operator._resume_lockout(_CODE)
    assert result["status"] == "ok", result
    assert captured, "the issuer published no resume envelope"
    return captured


def _deliver(envelope: dict[str, Any]) -> bool:
    """Deliver *envelope* to a locked-out receiver; True if it recovered."""
    robot = _mesh("robot-1")
    robot._estop_lockout.set()
    sample = MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(envelope).encode()
    robot._on_safety_resume(sample)
    return not robot._estop_lockout.is_set()


class TestClockSkewBlocksEstopRecovery:
    """A receiver behind the operator refuses a resume it should honour."""

    @pytest.fixture(autouse=True)
    def _code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)

    def test_a_receiver_within_the_forward_bound_recovers(self) -> None:
        """Control: with clocks in sync the correct code clears the lockout."""
        assert _deliver(_mint_resume()) is True

    def test_a_receiver_seconds_behind_the_operator_stays_locked_out(self, caplog: pytest.LogCaptureFixture) -> None:
        """One second past the forward bound is enough to refuse recovery."""
        skew = core._resume_forward_skew_s() + 1.0
        with caplog.at_level("WARNING"):
            recovered = _deliver(_mint_resume(issuer_clock_offset_s=skew))
        assert recovered is False, (
            f"a receiver {skew:.0f}s behind the operator accepted the resume; "
            "the forward-skew bound is what makes this refusal happen"
        )
        assert any("in future" in r.message for r in caplog.records), (
            "the refusal must say the envelope looked future-dated so an "
            f"operator can tell a clock problem from a bad code: {caplog.text}"
        )

    def test_retrying_does_not_help_because_every_envelope_is_refused(self) -> None:
        """The failure is not transient - a retry loop cannot recover the fleet."""
        skew = core._resume_forward_skew_s() + 1.0
        assert [_deliver(_mint_resume(issuer_clock_offset_s=skew)) for _ in range(3)] == [
            False,
            False,
            False,
        ]

    def test_widening_the_documented_bound_restores_recovery(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The documented remedy works: raise the bound and the same skew passes.

        This is what makes the knob worth documenting - it is the only in-process
        way out of a skew-induced lockout.
        """
        skew = core._resume_forward_skew_s() + 1.0
        monkeypatch.setenv("STRANDS_MESH_RESUME_FORWARD_SKEW_S", str(skew + 10.0))
        assert _deliver(_mint_resume(issuer_clock_offset_s=skew)) is True


class TestTheFreshnessBoundGovernsAReceiverAheadOfTheOperator:
    """The two bounds are not interchangeable: each governs one clock direction.

    ``_on_safety_resume`` refuses on ``envelope_t > now + forward_skew_s``
    (the envelope reads future-dated, which happens when the receiver's clock
    trails the operator's) and separately on ``now - envelope_t >
    freshness_window_s`` (the envelope reads stale, which happens when the
    receiver's clock *leads* the operator's). Documenting either bound against
    the wrong direction hands a locked-out operator a knob that cannot clear
    the refusal they are looking at, so the direction is pinned here.
    """

    @pytest.fixture(autouse=True)
    def _code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)

    def test_a_receiver_ahead_of_the_operator_is_refused_as_stale(self, caplog: pytest.LogCaptureFixture) -> None:
        """A negative issuer offset models a receiver whose clock leads."""
        lead = core._resume_freshness_window_s() + 1.0
        with caplog.at_level("WARNING"):
            recovered = _deliver(_mint_resume(issuer_clock_offset_s=-lead))
        assert recovered is False, (
            f"a receiver {lead:.0f}s ahead of the operator accepted the resume; "
            "the freshness window is what makes this refusal happen"
        )
        assert any("too old" in r.message for r in caplog.records), (
            "the refusal must say the envelope looked stale, which is the "
            f"direction STRANDS_MESH_RESUME_FRESHNESS_S governs: {caplog.text}"
        )

    def test_widening_the_freshness_window_recovers_a_receiver_that_leads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The documented remedy clears the refusal for the direction it names."""
        lead = core._resume_freshness_window_s() + 1.0
        monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", str(lead + 10.0))
        assert _deliver(_mint_resume(issuer_clock_offset_s=-lead)) is True

    def test_widening_the_freshness_window_does_not_recover_a_receiver_that_trails(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Why the direction matters: the wrong knob leaves the fleet stopped.

        A receiver *behind* the operator is refused by the forward-skew bound,
        so raising the freshness window -- however far -- changes nothing. This
        is the concrete cost of describing the freshness bound as the one that
        governs a trailing receiver.
        """
        trail = core._resume_forward_skew_s() + 1.0
        monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", "300")
        with caplog.at_level("WARNING"):
            recovered = _deliver(_mint_resume(issuer_clock_offset_s=trail))
        assert recovered is False, (
            "widening the freshness window recovered a trailing receiver; if "
            "that ever becomes true the documented remedy for each direction changes"
        )
        assert any("in future" in r.message for r in caplog.records), (
            "a trailing receiver must still be refused by the forward-skew "
            f"bound, not the freshness window: {caplog.text}"
        )


def _rendered_configuration() -> str:
    """``configuration.md`` with ``{{env_vars}}`` expanded by the shipped hook.

    The hook writes names as ``<code>NAME</code>``; the tags are folded to
    backticks so the same name regex reads the generated table and hand prose.
    """
    source = _CONFIGURATION.read_text(encoding="utf-8")
    module = docs_hook("env_vars")
    rendered = module.on_page_markdown(source, page=None, config=None, files=None)
    assert rendered != source, "docs/reference/configuration.md carries no {{env_vars}} token for the hook to expand"
    return re.sub(r"</?code>", "`", rendered)


def _env_table_rows() -> list[tuple[str, str]]:
    """Return ``(name_cell, meaning_cell)`` for every env-var row of the generated table.

    The table the hook renders is ``| variable | read in | default | meaning |``;
    the meaning cell is the one that may cite another variable.
    """
    rows = []
    for line in _rendered_configuration().splitlines():
        cells = [c.strip() for c in line.strip().split("|")[1:-1]]
        if len(cells) != 4:
            continue
        name_cell, _read_in, _default, meaning = cells
        if re.search(r"`(?:STRANDS|ZENOH)_[A-Z0-9_]+`", name_cell):
            rows.append((name_cell, meaning))
    return rows


def _safety_page_sentences(knob: str) -> list[str]:
    """Return every sentence on the safety page that names *knob*.

    ``docs/learn/mesh/safety-and-estop.md`` is where the resume gate is explained
    to an operator, so it is where each bound's clock direction must be stated.
    Sentences are split on a full stop followed by whitespace, with whitespace
    collapsed first so a re-wrap cannot move the claim out of reach.
    """
    text = " ".join(_SAFETY_PAGE.read_text(encoding="utf-8").split())
    sentences = [s for s in re.split(r"(?<=\.)\s+", text) if f"`{knob}`" in s]
    assert sentences, f"premise: no sentence on docs/learn/mesh/safety-and-estop.md names `{knob}`"
    return sentences


_DIRECTION = re.compile(r"\*?(ahead of|behind)\*? the operator")
_KNOB_MENTION = re.compile(r"`(STRANDS_MESH_[A-Z0-9_]+)`")


def _own_clause(sentence: str, knob: str) -> str:
    """The part of *sentence* that speaks about *knob*.

    From the knob's mention to the next mention of another ``STRANDS_MESH_``
    variable, or the end of the sentence. A sentence that names both bounds
    (``... older than A ..., and more than B ahead ...``) is thereby
    read once per bound, and a cross-reference to the sibling bound is not
    mistaken for a claim about this one.
    """
    mentions = list(_KNOB_MENTION.finditer(sentence))
    starts = [m.end() for m in mentions if m.group(1) == knob]
    if not starts:
        return ""
    start = starts[0]
    following = [m.start() for m in mentions if m.start() > start and m.group(1) != knob]
    return sentence[start : following[0] if following else len(sentence)]


def _documented_direction(knob: str) -> str:
    """The one clock direction the safety page attributes to *knob*.

    Every clause about the knob that states a direction must state the same
    one; a page saying both would leave the operator to guess.
    """
    sentences = _safety_page_sentences(knob)
    directions = {m.group(1).removesuffix(" of") for s in sentences for m in _DIRECTION.finditer(_own_clause(s, knob))}
    assert directions, (
        f"the safety page names {knob} but never says whether it catches a receiver 'ahead of' or "
        f"'behind' the operator, so the direction it claims cannot be read: {sentences}"
    )
    assert len(directions) == 1, f"the safety page attributes both directions to {knob}: {sentences}"
    return directions.pop()


class TestTheRecoveryKnobsAreDocumented:
    """The knobs a locked-out operator needs must be findable."""

    def test_every_knob_the_resume_freshness_gate_consults_is_documented(self) -> None:
        """Each bound the receiver enforces needs a row of its own.

        A knob that only exists in the source cannot be reached for by an
        operator whose fleet is already refusing every resume.
        """
        documented = {
            name for name_cell, _desc in _env_table_rows() for name in re.findall(r"`([A-Z_][A-Z0-9_]*)`", name_cell)
        }
        assert documented, "found no env-var rows in docs/reference/configuration.md; the scan is broken"
        required = (
            "STRANDS_MESH_OVERRIDE_CODE",
            "STRANDS_MESH_RESUME_FRESHNESS_S",
            "STRANDS_MESH_RESUME_FORWARD_SKEW_S",
        )
        missing = [name for name in required if name not in documented]
        assert not missing, (
            f"these govern whether a resume is accepted but have no configuration.md row: {missing}. "
            "An operator can only discover them once the fleet is already locked out."
        )

    def test_the_env_table_names_no_variable_it_never_lists(self) -> None:
        """A row that cites another variable implies the reader can look it up.

        Citing a name the table never lists sends the reader looking for a row
        that does not exist, which is worst on a safety knob a locked-out
        operator is trying to reach.
        """
        rows = _env_table_rows()
        assert rows, "found no env-var rows in docs/reference/configuration.md; the scan is broken"
        listed = {name for name_cell, _desc in rows for name in re.findall(r"`([A-Z_][A-Z0-9_]*)`", name_cell)}
        dangling = sorted(
            {
                (re.findall(r"`([A-Z_][A-Z0-9_]*)`", name_cell)[0], cited)
                for name_cell, desc in rows
                for cited in re.findall(r"`((?:STRANDS|ZENOH)_[A-Z0-9_]+)`", desc)
                if cited not in listed
            }
        )
        assert not dangling, (
            f"these env vars are cited in a description but have no row of their own (cited_by, missing): {dangling}"
        )

    def test_each_timestamp_bound_is_documented_against_the_clock_it_governs(self) -> None:
        """The rows must not swap the two directions.

        Behaviour tests above pin which bound refuses which skew; nothing
        otherwise ties that to the prose an operator actually reads, so a row
        can invert while the suite stays green.
        """
        assert _documented_direction("STRANDS_MESH_RESUME_FRESHNESS_S") == "ahead", (
            "the safety page's freshness sentence must attribute the stale-envelope lockout to a "
            "receiver whose clock is AHEAD of the operator -- that is the "
            "direction `now - envelope_t > freshness_window_s` trips on"
        )
        assert _documented_direction("STRANDS_MESH_RESUME_FORWARD_SKEW_S") == "behind", (
            "the safety page's forward-skew sentence must attribute its lockout to a receiver "
            "whose clock is BEHIND the operator"
        )

    def test_the_recovery_procedure_is_documented_beside_the_estop_call(self) -> None:
        """``docs/learn/mesh/safety-and-estop.md`` shows ``emergency_stop()``; it must show the way back."""
        mesh_doc = _SAFETY_PAGE.read_text(encoding="utf-8")
        assert "emergency_stop()" in mesh_doc, "premise: safety-and-estop.md documents emergency_stop"
        assert '"action": "resume"' in mesh_doc, (
            "safety-and-estop.md documents how to stop a fleet but not how to resume it"
        )
        for knob in ("STRANDS_MESH_OVERRIDE_CODE", "STRANDS_MESH_RESUME_FORWARD_SKEW_S"):
            assert knob in mesh_doc, f"safety-and-estop.md's recovery guidance omits {knob}"


#: Skew directions a receiver's clock can carry relative to the operator's, and
#: the ``issuer_clock_offset_s`` sign :func:`_mint_resume` needs for each. A
#: receiver *behind* the operator sees an envelope stamped in its own future, so
#: the issuer's clock runs ahead; a receiver *ahead* sees one already past.
_SKEW_SIGN = {"ahead": -1.0, "behind": +1.0}

#: The two bounds a skewed resume envelope can trip.
_BOUNDS = (
    "STRANDS_MESH_RESUME_FRESHNESS_S",
    "STRANDS_MESH_RESUME_FORWARD_SKEW_S",
)

#: Skew (seconds) past both defaults, so a receiver at this offset is refused
#: until the bound governing its own direction is widened.
_SKEW_S = 61.0


def _recovers_when_widened(knob: str, direction: str, monkeypatch: pytest.MonkeyPatch) -> bool:
    """Widen *knob*, deliver a resume to a receiver skewed *direction*; recovered?"""
    monkeypatch.setenv(knob, str(_SKEW_S + 30.0))
    try:
        return _deliver(_mint_resume(issuer_clock_offset_s=_SKEW_SIGN[direction] * _SKEW_S))
    finally:
        monkeypatch.delenv(knob, raising=False)


def _bound_governing(direction: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """The one bound in :data:`_BOUNDS` whose widening recovers *direction*.

    Measured against the shipped receiver rather than recorded here, so if which
    bound catches which direction ever changes, the rows are re-graded against
    the new behaviour instead of staying pinned to a claim that has gone stale.
    """
    clearing = [knob for knob in _BOUNDS if _recovers_when_widened(knob, direction, monkeypatch)]
    assert len(clearing) == 1, (
        f"expected exactly one bound to clear a receiver {direction} the operator, got {clearing}. "
        "The rows describe a one-bound-per-direction split; if that is no longer how the receiver "
        "behaves the rows need rewriting, not this assertion relaxing."
    )
    return clearing[0]


def _own_direction_claim(knob: str) -> str:
    """The skew direction the safety page says *knob* governs; see :func:`_documented_direction`."""
    return _documented_direction(knob)


class TestTheDocumentedDirectionIsGradedAgainstTheReceiver:
    """Grade both rows against the receiver rather than against recorded text.

    ``test_each_timestamp_bound_is_documented_against_the_clock_it_governs``
    pins each row's direction word to the direction that bound catches today.
    That catches a row being edited to the wrong direction, but not the mirror
    case: swap which bound the receiver applies to which sign of skew and the
    pinned words silently become wrong again, in the other direction, with the
    suite green. Deriving both sides means neither can drift alone.
    """

    @pytest.fixture(autouse=True)
    def _code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)

    @pytest.mark.parametrize("knob", _BOUNDS)
    def test_each_row_names_the_direction_its_own_bound_governs(
        self, knob: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The direction a row claims must be the one its bound really catches."""
        documented = _own_direction_claim(knob)
        enforced = _bound_governing(documented, monkeypatch)
        assert enforced == knob, (
            f"{knob}'s safety page sentence says it governs a receiver {documented} the operator, but a "
            f"receiver {documented} the operator is refused until {enforced} is widened - widening "
            f"{knob} leaves it locked out. Name the direction this bound catches."
        )

    @pytest.mark.parametrize("knob", _BOUNDS)
    def test_widening_the_bound_a_row_names_clears_the_skew_it_names(
        self, knob: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Follow each row literally: widen its bound, in the case it describes.

        Both rows now tell a locked-out operator to widen the bound for the skew
        they are looking at. Executing that instruction grades the remedy rather
        than the wording, so a row can be refused for prescribing a knob that
        does not clear the refusal it names even if its direction word is right.
        """
        documented = _own_direction_claim(knob)
        assert _recovers_when_widened(knob, documented, monkeypatch) is True, (
            f"{knob}'s safety page sentence prescribes widening it for a receiver {documented} the "
            f"operator, but doing exactly that left the receiver locked out. A remedy that "
            "does not clear the refusal it names is worse than none - the fleet is already stopped."
        )
