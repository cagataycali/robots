"""The two ``IsaacConfig`` boolean environment switches are held to one vocabulary.

``STRANDS_ISAAC_HEADLESS`` and ``STRANDS_ISAAC_RTX_PATHTRACING`` are both
documented as two-sided switches -- ``docs/simulation/isaac.md`` and the README
each said, until the change this module tests rewrote them to enumerate the four
pairs below, "Truthy (``1``/``true``/``yes``) forces headless; falsy forces a
window" -- but only the truthy side was enumerated. Everything else fell through
to the falsy branch, so a spelling that means *on* forced the outcome the
variable exists to prevent:

===========================  ====================  ====================
``STRANDS_ISAAC_HEADLESS``   before                after
===========================  ====================  ====================
``"on"``                     windowed             headless
``"enabled"``                windowed             refused
``" true"`` (whitespace)     windowed             headless
``""`` (set but empty)       windowed             the field is kept
``"false"`` / ``"0"``        windowed             windowed
``"true"`` / ``"1"``         headless             headless
===========================  ====================  ====================

The headline test is stated over spellings rather than over types: no spelling
that a reader would call *on* may resolve to *off*. It may be refused -- an
unrecognized spelling is not a documented side -- but it must never silently
land on the opposite side of the switch it names.

Solver-free: every assertion here constructs an ``IsaacConfig`` only, which
runs before Isaac Sim's ``SimulationApp`` exists and needs no ``isaacsim``.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_inherited_isaac_env(monkeypatch):
    """Neither switch may be inherited from the runner that runs these tests."""
    monkeypatch.delenv("STRANDS_ISAAC_HEADLESS", raising=False)
    monkeypatch.delenv("STRANDS_ISAAC_RTX_PATHTRACING", raising=False)


def _config(**kwargs):
    from strands_robots.simulation.isaac.config import IsaacConfig

    return IsaacConfig(**kwargs)


def _vocabularies():
    from strands_robots.simulation.isaac.config import ENV_SWITCH_OFF, ENV_SWITCH_ON

    return ENV_SWITCH_ON, ENV_SWITCH_OFF


# Spellings a reader would call "on" that are in neither documented vocabulary.
# Each one resolved to windowed before this change.
UNLISTED_ON_SPELLINGS = ("enabled", "y", "t", "Y", "yeah", "affirmative", "2")

# Spellings that carry no side at all. These were windowed too.
UNLISTED_MEANINGLESS_SPELLINGS = ("maybe", "banana", "-1", "None", "null")


class TestTheDocumentedVocabularyResolves:
    """Each listed spelling selects the side it names, on both switches."""

    def test_every_on_spelling_forces_headless(self, monkeypatch):
        on, _ = _vocabularies()
        for spelling in on:
            monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", spelling)
            assert _config(headless=False).headless is True, spelling

    def test_every_off_spelling_forces_windowed(self, monkeypatch):
        _, off = _vocabularies()
        for spelling in off:
            monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", spelling)
            assert _config(headless=True).headless is False, spelling

    def test_the_two_vocabularies_are_symmetric_pairs(self):
        """The defect was an enumerated on side against an open-ended off side."""
        on, off = _vocabularies()
        assert len(on) == len(off)
        assert set(on).isdisjoint(off)
        assert ("on" in on) and ("off" in off)

    def test_case_is_ignored_on_both_sides(self, monkeypatch):
        for spelling, expected in (("TRUE", True), ("On", True), ("FALSE", False), ("Off", False)):
            monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", spelling)
            assert _config().headless is expected, spelling

    def test_surrounding_whitespace_is_ignored(self, monkeypatch):
        """A trailing newline is what a heredoc or a file-sourced value carries."""
        for spelling in (" true", "true\n", "\ttrue\t", " 1 "):
            monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", spelling)
            assert _config(headless=False).headless is True, spelling

    def test_the_resolved_value_is_a_real_bool(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "yes")
        assert _config().headless is True
        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "no")
        assert _config().headless is False


class TestNoSpellingThatMeansOnResolvesToOff:
    """The headline property, stated over spellings rather than over types."""

    def test_an_unlisted_on_spelling_is_never_read_as_off(self, monkeypatch):
        for spelling in UNLISTED_ON_SPELLINGS:
            monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", spelling)
            with pytest.raises(ValueError, match="not a recognized switch value"):
                _config()

    def test_a_meaningless_spelling_is_never_read_as_off(self, monkeypatch):
        for spelling in UNLISTED_MEANINGLESS_SPELLINGS:
            monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", spelling)
            with pytest.raises(ValueError, match="not a recognized switch value"):
                _config()

    def test_both_parties_asking_for_headless_cannot_produce_a_window(self, monkeypatch):
        """The measured worst case: a caller passing headless=True plus an operator
        setting the switch to an opt-in spelling resolved to a window."""
        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "on")
        assert _config(headless=True).headless is True

    def test_the_off_side_stayed_intact_for_every_spelling_it_already_had(self, monkeypatch):
        """Closing the off side is the cost of making the on side whole, so the
        spellings that already meant off must keep meaning off rather than start
        raising."""
        for spelling in ("false", "0", "no", "off"):
            monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", spelling)
            assert _config().headless is False, spelling


class TestAnEmptyValueIsAbsentRatherThanOff:
    """A set-but-empty variable is the shell's spelling of unset."""

    def test_an_empty_value_keeps_the_field(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "")
        assert _config().headless is True
        assert _config(headless=False).headless is False

    def test_a_whitespace_only_value_keeps_the_field(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "   ")
        assert _config(headless=False).headless is False

    def test_an_unset_variable_keeps_the_field(self):
        assert _config().headless is True
        assert _config(headless=False).headless is False

    def test_an_empty_value_does_not_force_pathtracing(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ISAAC_RTX_PATHTRACING", "")
        assert _config(render_mode="rtx_realtime").render_mode == "rtx_realtime"


class TestTheRefusalNamesBothVocabularies:
    """A refusal a caller cannot act on is a different defect."""

    def test_the_message_names_the_variable_and_the_value(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "enabled")
        with pytest.raises(ValueError) as excinfo:
            _config()
        message = str(excinfo.value)
        assert "STRANDS_ISAAC_HEADLESS" in message
        assert "'enabled'" in message

    def test_the_message_lists_both_sides(self, monkeypatch):
        on, off = _vocabularies()
        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "maybe")
        with pytest.raises(ValueError) as excinfo:
            _config()
        message = str(excinfo.value)
        for spelling in on + off:
            assert repr(spelling) in message, spelling

    def test_the_message_says_empty_and_unset_keep_the_field(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "maybe")
        with pytest.raises(ValueError) as excinfo:
            _config()
        assert "unset or empty" in str(excinfo.value)

    def test_the_message_states_why_it_refuses_instead_of_reading_off(self, monkeypatch):
        """Without the reason, refusing looks stricter than the alternative
        rather than the only reading that is not a silent inversion."""
        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "maybe")
        with pytest.raises(ValueError) as excinfo:
            _config()
        assert "would otherwise resolve to off" in str(excinfo.value)

    def test_the_refusal_names_the_pathtracing_variable_when_that_is_the_one_set(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ISAAC_RTX_PATHTRACING", "enabled")
        with pytest.raises(ValueError) as excinfo:
            _config()
        assert "STRANDS_ISAAC_RTX_PATHTRACING" in str(excinfo.value)
        assert "STRANDS_ISAAC_HEADLESS" not in str(excinfo.value)


class TestThePathtracingSwitchIsHeldToTheSameVocabulary:
    """The neighbouring read did not invert, but it did silently ignore."""

    def test_an_on_spelling_forces_pathtracing(self, monkeypatch):
        on, _ = _vocabularies()
        for spelling in on:
            monkeypatch.setenv("STRANDS_ISAAC_RTX_PATHTRACING", spelling)
            assert _config().render_mode == "rtx_pathtracing", spelling

    def test_an_off_spelling_leaves_the_render_mode_alone(self, monkeypatch):
        """This switch names one mode, so its off side is "do not force it"
        rather than a second mode to select."""
        _, off = _vocabularies()
        for spelling in off:
            monkeypatch.setenv("STRANDS_ISAAC_RTX_PATHTRACING", spelling)
            assert _config(render_mode="rtx_realtime").render_mode == "rtx_realtime", spelling
            assert _config().render_mode == "headless", spelling

    def test_an_unlisted_spelling_is_refused_rather_than_ignored(self, monkeypatch):
        for spelling in UNLISTED_ON_SPELLINGS + UNLISTED_MEANINGLESS_SPELLINGS:
            monkeypatch.setenv("STRANDS_ISAAC_RTX_PATHTRACING", spelling)
            with pytest.raises(ValueError, match="not a recognized switch value"):
                _config()

    def test_both_switches_can_be_set_together(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "off")
        monkeypatch.setenv("STRANDS_ISAAC_RTX_PATHTRACING", "on")
        config = _config()
        assert config.headless is False
        assert config.render_mode == "rtx_pathtracing"


class TestTheSharedOptInReaderWasNotReusedForAMeasuredReason:
    """Measured rather than asserted, so the choice stays checkable."""

    def test_env_flag_cannot_express_the_third_outcome(self, monkeypatch):
        """``safe_output.env_flag`` returns ``bool``, so "set to off" and
        "unset" are the same answer -- which is why it cannot back a switch
        that has to force the non-default side."""
        from strands_robots.simulation.safe_output import env_flag

        monkeypatch.delenv("A_SWITCH", raising=False)
        assert env_flag("A_SWITCH") is False
        monkeypatch.setenv("A_SWITCH", "false")
        assert env_flag("A_SWITCH") is False

    def test_env_switch_distinguishes_the_three_outcomes(self, monkeypatch):
        from strands_robots.simulation.isaac.config import _env_switch

        monkeypatch.delenv("A_SWITCH", raising=False)
        assert _env_switch("A_SWITCH") is None
        monkeypatch.setenv("A_SWITCH", "false")
        assert _env_switch("A_SWITCH") is False
        monkeypatch.setenv("A_SWITCH", "true")
        assert _env_switch("A_SWITCH") is True

    def test_env_flag_keeps_its_one_sided_contract(self, monkeypatch):
        """The opt-in reader is not widened by this change: an unrecognized
        spelling there still means "not opted in" and raises nothing, because a
        permission grant has no off side to confuse it with."""
        from strands_robots.simulation.safe_output import env_flag

        monkeypatch.setenv("A_SWITCH", "enabled")
        assert env_flag("A_SWITCH") is False


class TestNeighbouringSurfacesStayOutOfScope:
    """Boundary pins. Replace rather than delete these if the scope moves."""

    def test_the_environment_still_outranks_the_field(self, monkeypatch):
        """Precedence is untouched here. The documentation no longer contradicts
        itself about it -- ``docs/simulation/isaac.md`` and both README tables
        now state it per variable and link #2062 -- but *which* direction the two
        switches should have is still the open contract decision there, so this
        is pinned rather than resolved."""
        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "false")
        assert _config(headless=True).headless is False

    def test_the_sibling_url_variable_keeps_the_opposite_precedence(self, monkeypatch):
        """Same function, other direction: the field wins for ``nucleus_url``.
        Pinned so the contradiction cannot be resolved by accident."""
        monkeypatch.setenv("STRANDS_ISAAC_NUCLEUS_URL", "omniverse://from-env")
        assert _config().nucleus_url == "omniverse://from-env"
        assert _config(nucleus_url="omniverse://explicit").nucleus_url == "omniverse://explicit"

    def test_the_pathtracing_switch_also_outranks_its_field(self, monkeypatch):
        """The third variable, and the one whose precedence nothing pinned.

        Its *off* side is already held against an explicit field by
        ``TestThePathtracingSwitchIsHeldToTheSameVocabulary``. The *on* side was
        only ever exercised against the default ``render_mode``, so this switch
        was the one of the three that could reverse to field-wins undetected --
        measured on ``24766c3`` by gating the assignment on
        ``self.render_mode == "headless"``: all 406 tests under
        ``tests/simulation/isaac`` still passed, as did every test module that
        mentions ``render_mode`` at all.

        Stated over the whole enumeration rather than one representative value:
        no mode a caller can pass explicitly survives the switch.
        """
        from strands_robots.simulation.isaac.config import RENDER_MODES

        monkeypatch.setenv("STRANDS_ISAAC_RTX_PATHTRACING", "on")
        for mode in RENDER_MODES:
            assert _config(render_mode=mode).render_mode == "rtx_pathtracing", mode

    def test_the_headless_field_itself_is_not_type_checked(self):
        """A non-bool passed directly is still accepted. That is the argument
        domain rather than the environment vocabulary, and it is a separate
        change."""
        assert _config(headless="false").headless == "false"


class TestNoIsaacEnvSwitchSurfaceDrifts:
    """A new switch in this module cannot skip the shared reader."""

    def _source(self):
        from pathlib import Path

        from strands_robots.simulation.isaac import config

        return Path(config.__file__).read_text(encoding="utf-8")

    def test_every_isaac_switch_is_read_through_env_switch(self):
        source = self._source()
        assert source.count('_env_switch("STRANDS_ISAAC_') == 2

    def test_only_the_url_variable_is_read_directly(self):
        """``nucleus_url`` is a string rather than a switch, so it is the one
        legitimate direct read. Any other would be a switch bypassing the
        vocabulary."""
        source = self._source()
        direct = [line.strip() for line in source.splitlines() if 'os.environ.get("STRANDS_ISAAC' in line]
        assert direct == ['self.nucleus_url = os.environ.get("STRANDS_ISAAC_NUCLEUS_URL")']

    def test_no_inline_truthiness_vocabulary_survives(self):
        """The defect's shape was an inline truthy list. One is left, inside the
        reader, expressed as the two named vocabularies rather than a literal."""
        source = self._source()
        assert '.lower() in ("true"' not in source
        assert '.lower() in ("1"' not in source
