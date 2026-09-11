# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An Isaac integer knob that falls back to its default says so.

:func:`~strands_robots.simulation.isaac.simulation._env_int` resolves three
step counts - ``STRANDS_ISAAC_CAMERA_WARMUP_STEPS``, ``SO101_RECORD_CONVERGE``
and ``SO101_IDLE_CONVERGE`` - and substituted the default for every value it
could not honor without a word, while :func:`_env_float` eight lines below it
reports each of its own rejections and states why: a knob whose effect is only
visible several calls away leaves the operator nothing to correct against when
the substitution is silent. Measured on the two resolvers side by side, same
raw value, same process:

==========  =============  ======================  =============  ============
raw         ``_env_int``   int log                 ``_env_float`` float log
==========  =============  ======================  =============  ============
``3O``      10             (silent)                1.0            not a number
``10.0``    10             (silent)                10.0           (honored)
``0``       10             (silent)                1.0            must be > 0
``-3``      10             (silent)                1.0            must be > 0
``"  "``    10             (silent)                1.0            (silent)
``5``       5              (silent)                5.0            (silent)
==========  =============  ======================  =============  ============

The first four int rows are the defect. ``STRANDS_ISAAC_CAMERA_WARMUP_STEPS``
is the count of render-bearing steps ``add_camera`` takes so a new RTX camera's
first ``get_rgba()`` is a real frame rather than the pipeline's empty buffer,
and its documentation says to raise it on a slow GPU - so an operator who did
and mistyped it got the default count and an empty first frame, with nothing
naming the variable. The blank and unset rows are controls: not setting a knob
is not a misconfiguration and must stay quiet, as it does for the float.

The accepted domain is deliberately unchanged. ``10.0`` is still refused, as
:func:`~strands_robots.mesh.core._parse_positive_int_env` refuses it, because
this is a whole-number knob read from a string; what changed is that the
refusal is now reported. Solver-free: the resolver reads the environment and
the logger only, so no Isaac Sim Kit runtime is touched.
"""

from __future__ import annotations

import pytest

import strands_robots.simulation.isaac.simulation as isaac_module
from strands_robots.simulation.isaac.simulation import _env_float, _env_int

WARMUP_ENV = "STRANDS_ISAAC_CAMERA_WARMUP_STEPS"
DEFAULT_WARMUP = 10

NOT_A_WHOLE_NUMBER = ["3O", "ten", "1e2", "10.0", "inf", "nan"]
NOT_POSITIVE = ["0", "-3", "-0"]
BLANK = ["", "  ", "\t"]


@pytest.fixture(autouse=True)
def _clear_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    """The knob must not be inherited from the runner running these tests."""
    monkeypatch.delenv(WARMUP_ENV, raising=False)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


class TestAnUnusableCountIsReportedBeforeItFallsBack:
    """The defect: the default applied in silence."""

    @pytest.mark.parametrize("raw", NOT_A_WHOLE_NUMBER)
    def test_a_value_int_cannot_parse_names_the_variable_and_the_value(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
    ) -> None:
        monkeypatch.setenv(WARMUP_ENV, raw)
        with caplog.at_level("WARNING", logger=isaac_module.__name__):
            assert _env_int(WARMUP_ENV, DEFAULT_WARMUP) == DEFAULT_WARMUP
        messages = _warnings(caplog)
        assert len(messages) == 1, messages
        assert WARMUP_ENV in messages[0] and repr(raw) in messages[0], messages

    @pytest.mark.parametrize("raw", NOT_POSITIVE)
    def test_a_non_positive_count_names_the_bound(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
    ) -> None:
        monkeypatch.setenv(WARMUP_ENV, raw)
        with caplog.at_level("WARNING", logger=isaac_module.__name__):
            assert _env_int(WARMUP_ENV, DEFAULT_WARMUP) == DEFAULT_WARMUP
        messages = _warnings(caplog)
        assert len(messages) == 1, messages
        assert WARMUP_ENV in messages[0] and "> 0" in messages[0], messages

    @pytest.mark.parametrize("raw", NOT_A_WHOLE_NUMBER + NOT_POSITIVE)
    def test_the_report_states_the_value_that_is_used_instead(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
    ) -> None:
        """Naming the substitute is what lets the operator predict the run."""
        monkeypatch.setenv(WARMUP_ENV, raw)
        with caplog.at_level("WARNING", logger=isaac_module.__name__):
            _env_int(WARMUP_ENV, 7)
        assert any("using 7" in m for m in _warnings(caplog)), _warnings(caplog)


class TestTheAcceptedDomainIsUnchanged:
    """Over-reach controls: only the silence moved."""

    @pytest.mark.parametrize(("raw", "expected"), [("5", 5), (" 7 ", 7), ("+12", 12), ("1", 1)])
    def test_a_positive_whole_number_is_honored_without_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str, expected: int
    ) -> None:
        monkeypatch.setenv(WARMUP_ENV, raw)
        with caplog.at_level("WARNING", logger=isaac_module.__name__):
            assert _env_int(WARMUP_ENV, DEFAULT_WARMUP) == expected
        assert _warnings(caplog) == []

    @pytest.mark.parametrize("raw", BLANK)
    def test_a_blank_value_is_unset_and_stays_quiet(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
    ) -> None:
        """Not setting a knob is not a misconfiguration."""
        monkeypatch.setenv(WARMUP_ENV, raw)
        with caplog.at_level("WARNING", logger=isaac_module.__name__):
            assert _env_int(WARMUP_ENV, DEFAULT_WARMUP) == DEFAULT_WARMUP
        assert _warnings(caplog) == []

    def test_an_unset_variable_resolves_to_the_default_without_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger=isaac_module.__name__):
            assert _env_int(WARMUP_ENV, DEFAULT_WARMUP) == DEFAULT_WARMUP
        assert _warnings(caplog) == []

    @pytest.mark.parametrize("raw", ["10.0", "1e2"])
    def test_a_float_spelling_is_still_refused_not_widened(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        """A whole-number knob read from a string keeps ``int()``'s domain, as
        the mesh's ``_parse_positive_int_env`` does; only the report is new.
        """
        monkeypatch.setenv(WARMUP_ENV, raw)
        assert _env_int(WARMUP_ENV, DEFAULT_WARMUP) == DEFAULT_WARMUP


class TestTheTwoResolversReportTheSameClassOfInput:
    """The two knob resolvers in one module cannot diverge on whether a
    rejection is reported: a value neither can honor is reported by both, and
    a blank neither treats as set is reported by neither.
    """

    @pytest.mark.parametrize("raw", ["3O", "0", "-3"])
    def test_a_value_both_refuse_is_reported_by_both(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
    ) -> None:
        monkeypatch.setenv(WARMUP_ENV, raw)
        with caplog.at_level("WARNING", logger=isaac_module.__name__):
            _env_int(WARMUP_ENV, DEFAULT_WARMUP)
            _env_float(WARMUP_ENV, 1.0)
        assert len(_warnings(caplog)) == 2, _warnings(caplog)

    @pytest.mark.parametrize("raw", BLANK)
    def test_a_blank_is_reported_by_neither(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
    ) -> None:
        monkeypatch.setenv(WARMUP_ENV, raw)
        with caplog.at_level("WARNING", logger=isaac_module.__name__):
            _env_int(WARMUP_ENV, DEFAULT_WARMUP)
            _env_float(WARMUP_ENV, 1.0)
        assert _warnings(caplog) == []
