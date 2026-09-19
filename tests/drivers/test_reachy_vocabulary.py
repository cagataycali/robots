"""The pure helpers behind the Reachy Mini's expressive vocabulary.

Every function in :mod:`strands_robots.drivers.reachy_vocabulary` is a pure
function of its arguments, so each fact here is graded with no driver, no
transport and no daemon: an emotion word lands on a library move, a ``goto``
body arrives in the daemon's units, the discovery list reads the environment
the documented way, a daemon status is judged usable or not, and a volume word
becomes a level.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.drivers import reachy_vocabulary as vocab

_CATALOGUE = ["cheerful1", "curious1", "no1", "yes1", "displeased1", "go_away1", "dance1", "sad1", "enthusiastic2"]


class TestAPlainEmotionWordResolvesToALibraryMove:
    """``resolve_move_name``: exact, alias, ``+1``, unique prefix - in that order."""

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            ("cheerful1", "cheerful1"),  # exact
            ("happy", "cheerful1"),  # alias table
            ("HAPPY ", "cheerful1"),  # case and whitespace tolerated
            ("no", "no1"),  # alias beats the +1 rule but agrees with it
            ("curious", "curious1"),  # name + "1"
            ("go away", "go_away1"),  # spaces become underscores
            ("go-away", "go_away1"),  # hyphens too
            ("displeased", "displeased1"),  # unique prefix
        ],
    )
    def test_each_resolution_rule_lands_on_a_real_move(self, requested: str, expected: str) -> None:
        assert vocab.resolve_move_name(requested, _CATALOGUE) == expected

    def test_a_word_nothing_matches_is_none_not_a_guess(self) -> None:
        assert vocab.resolve_move_name("juggling", _CATALOGUE) is None

    def test_an_empty_request_is_none(self) -> None:
        assert vocab.resolve_move_name("", _CATALOGUE) is None
        assert vocab.resolve_move_name("   ", _CATALOGUE) is None

    def test_a_stale_alias_is_not_returned_when_the_catalogue_lacks_it(self) -> None:
        # "angry" -> rage1 in the table; a library without rage1 must not be sent it.
        assert vocab.resolve_move_name("angry", ["cheerful1"]) is None

    def test_the_alias_table_alone_answers_when_the_catalogue_is_empty(self) -> None:
        # No catalogue read (daemon could not list) must not turn every plain
        # word into a refusal - that is the failure the table exists to end.
        assert vocab.resolve_move_name("happy", []) == "cheerful1"
        assert vocab.resolve_move_name("juggling", []) is None

    def test_every_alias_target_is_a_bare_url_path_segment(self) -> None:
        # The driver interpolates the resolved name into a URL path; the table
        # must never be able to manufacture a dot segment.
        import re

        segment = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
        assert all(segment.fullmatch(target) for target in vocab.EMOTION_ALIASES.values())


class TestAGotoBodyArrivesInTheDaemonsUnits:
    """Degrees and millimetres in, radians and metres out, only the groups named."""

    def test_a_whole_pose_converts_every_axis(self) -> None:
        body = vocab.goto_body(
            head={"pitch": 15.0, "roll": -10.0, "yaw": 30.0, "x": 5.0, "y": -2.5, "z": 10.0},
            body_yaw=45.0,
            antennas=(20.0, -20.0),
            duration=0.8,
            interpolation="minjerk",
        )
        assert body["duration"] == 0.8
        assert body["interpolation"] == "minjerk"
        assert body["head_pose"]["pitch"] == pytest.approx(math.radians(15.0))
        assert body["head_pose"]["roll"] == pytest.approx(math.radians(-10.0))
        assert body["head_pose"]["yaw"] == pytest.approx(math.radians(30.0))
        assert body["head_pose"]["x"] == pytest.approx(0.005)
        assert body["head_pose"]["y"] == pytest.approx(-0.0025)
        assert body["head_pose"]["z"] == pytest.approx(0.010)
        assert body["body_yaw"] == pytest.approx(math.radians(45.0))
        assert body["antennas"] == [pytest.approx(math.radians(20.0)), pytest.approx(math.radians(-20.0))]

    def test_only_the_groups_named_are_sent(self) -> None:
        body = vocab.goto_body(head=None, body_yaw=None, antennas=(10.0, 10.0), duration=0.5, interpolation="linear")
        assert set(body) == {"duration", "interpolation", "antennas"}

    def test_an_absent_head_axis_is_zero_because_the_command_is_a_whole_pose(self) -> None:
        body = vocab.goto_body(
            head={"pitch": 10.0}, body_yaw=None, antennas=None, duration=1.0, interpolation="minjerk"
        )
        assert body["head_pose"]["yaw"] == 0.0
        assert body["head_pose"]["x"] == 0.0


class TestAGotoRequestIsRefusedByName:
    """``goto_body_error`` names the one field that is wrong."""

    def _error(self, **overrides: object) -> str | None:
        kwargs: dict[str, object] = {
            "head": {"pitch": 0.0},
            "body_yaw": None,
            "antennas": None,
            "duration": 1.0,
            "interpolation": "minjerk",
            "context": "goto",
        }
        kwargs.update(overrides)
        return vocab.goto_body_error(**kwargs)  # type: ignore[arg-type]

    def test_a_valid_request_has_no_error(self) -> None:
        assert self._error() is None

    def test_nothing_to_move_is_refused_rather_than_sent_as_a_no_op(self) -> None:
        reason = self._error(head=None)
        assert reason is not None and "nothing to move" in reason

    @pytest.mark.parametrize("duration", [0.05, 11.0, float("nan"), float("inf"), True, "1"])
    def test_a_duration_outside_the_window_or_not_a_number_is_refused(self, duration: object) -> None:
        reason = self._error(duration=duration)
        assert reason is not None and "duration" in reason

    def test_an_unknown_interpolation_is_refused_naming_the_admitted_set(self) -> None:
        reason = self._error(interpolation="bouncy")
        assert reason is not None and "bouncy" in reason and "minjerk" in reason

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_a_translation_past_the_platforms_travel_is_refused(self, axis: str) -> None:
        reason = self._error(head={axis: 25.5})
        assert reason is not None and axis in reason and "25 mm" in reason
        assert self._error(head={axis: 25.0}) is None

    def test_a_non_finite_translation_is_refused(self) -> None:
        reason = self._error(head={"x": float("nan")})
        assert reason is not None and "finite" in reason

    def test_an_antenna_past_its_travel_is_refused_by_side(self) -> None:
        reason = self._error(head=None, antennas=(151.0, 0.0))
        assert reason is not None and "antenna_right" in reason
        reason = self._error(head=None, antennas=(0.0, -151.0))
        assert reason is not None and "antenna_left" in reason
        assert self._error(head=None, antennas=(150.0, -150.0)) is None

    def test_an_antenna_pair_of_the_wrong_length_is_refused(self) -> None:
        reason = self._error(head=None, antennas=(1.0,))  # type: ignore[arg-type]
        assert reason is not None and "pair" in reason


class TestDiscoveryReadsTheEnvironmentTheDocumentedWay:
    """``discovery_candidates``: REACHY_HOST pins one address; otherwise the two defaults."""

    def test_without_the_environment_the_two_default_hosts_are_probed_in_order(self) -> None:
        assert vocab.discovery_candidates(8000, environ={}) == [("localhost", 8000), ("reachy-mini.local", 8000)]

    def test_the_api_port_argument_is_the_port_for_every_default_host(self) -> None:
        assert vocab.discovery_candidates(9100, environ={}) == [("localhost", 9100), ("reachy-mini.local", 9100)]

    def test_a_named_host_replaces_the_whole_list(self) -> None:
        assert vocab.discovery_candidates(8000, environ={"REACHY_HOST": "reachy-b.local"}) == [("reachy-b.local", 8000)]

    def test_a_host_with_its_own_suffix_keeps_that_port(self) -> None:
        assert vocab.discovery_candidates(8000, environ={"REACHY_HOST": "10.0.0.7:8001", "REACHY_PORT": "9000"}) == [
            ("10.0.0.7", 8001)
        ]

    def test_reachy_port_supplies_the_port_for_a_bare_host(self) -> None:
        assert vocab.discovery_candidates(8000, environ={"REACHY_HOST": "10.0.0.7", "REACHY_PORT": "9000"}) == [
            ("10.0.0.7", 9000)
        ]

    def test_reachy_port_alone_reprices_the_default_hosts(self) -> None:
        assert vocab.discovery_candidates(8000, environ={"REACHY_PORT": "9000"}) == [
            ("localhost", 9000),
            ("reachy-mini.local", 9000),
        ]

    def test_a_blank_host_is_the_same_as_none(self) -> None:
        assert vocab.discovery_candidates(8000, environ={"REACHY_HOST": "  "}) == vocab.discovery_candidates(
            8000, environ={}
        )


class TestADaemonStatusIsJudgedUsableOrNot:
    """``daemon_answers``: the desktop Lite daemon that found no robot is not a robot."""

    def test_a_running_daemon_answers(self) -> None:
        assert vocab.daemon_answers({"state": "running", "wireless_version": True, "error": None})

    def test_a_status_without_a_state_field_still_answers(self) -> None:
        # Older payloads and test doubles carry only the variant flag.
        assert vocab.daemon_answers({"wireless_version": False})

    def test_a_daemon_reporting_its_own_error_does_not(self) -> None:
        # Measured on a Mac with other USB-serial devices attached: the desktop
        # Lite daemon on :8000 answers with this and serves no robot.
        assert not vocab.daemon_answers(
            {"state": "error", "wireless_version": False, "error": "Multiple Reachy Mini serial ports found"}
        )

    def test_a_state_of_error_alone_does_not(self) -> None:
        assert not vocab.daemon_answers({"state": "error", "error": None})

    def test_the_transports_failure_envelope_does_not(self) -> None:
        assert not vocab.daemon_answers({"error": "Connection refused"})

    @pytest.mark.parametrize("body", [None, [], "running", 200])
    def test_a_body_that_is_not_an_object_does_not(self, body: object) -> None:
        assert not vocab.daemon_answers(body)


class TestAVolumeRequestBecomesALevel:
    """``resolve_volume_level``: numbers, words, relative words, refusals."""

    @pytest.mark.parametrize(
        ("level", "current", "expected"),
        [
            (40, None, 40),
            (40.4, None, 40),
            ("40", None, 40),
            ("40%", None, 40),
            ("silent", None, 0),
            ("MAX", None, 100),
            ("normal", None, 60),
            ("quieter", 60, 30),
            ("louder", 60, 80),
            ("louder", 90, "outside 0-100"),
        ],
    )
    def test_each_spelling(self, level: object, current: int | None, expected: object) -> None:
        result = vocab.resolve_volume_level(level, current)
        if isinstance(expected, str):
            assert isinstance(result, str) and expected in result
        else:
            assert result == expected

    def test_a_relative_word_without_a_current_level_is_refused_not_guessed(self) -> None:
        result = vocab.resolve_volume_level("quieter", None)
        assert isinstance(result, str) and "relative" in result

    @pytest.mark.parametrize("level", [True, float("nan"), "loudish", 101, -1])
    def test_a_value_outside_the_domain_is_refused(self, level: object) -> None:
        assert isinstance(vocab.resolve_volume_level(level, 50), str)

    @pytest.mark.parametrize(
        "level",
        ["inf", "infinity", "-inf", "1e999", "nan", "INF%", float("inf"), float("-inf")],
    )
    def test_a_non_finite_spelling_is_refused_not_raised(self, level: object) -> None:
        """``float`` accepts every spelling here, and ``round`` raises on an infinity.

        The string branch used to catch ``ValueError`` alone, so ``"infinity"``
        escaped as ``OverflowError`` through the tool's no-raise contract; the
        ``"nan"`` sibling was saved only because ``round(nan)`` happens to raise
        ``ValueError``. Both spellings, and the numeric infinities the other
        branch already refused, get the one finiteness reason.
        """
        result = vocab.resolve_volume_level(level, 50)
        assert isinstance(result, str)
        assert "finite" in result
