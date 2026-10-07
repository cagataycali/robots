"""Regression: the bare-allowlist warning says what the operator set, not what the gate probed.

The once-per-spelling WARNING about a bare ``STRANDS_ROBOT_COMMAND_ALLOW``
entry fired from inside ``allow_match``, and ``gate_motion`` probes that
matcher with one-entry sets that are not the operator's allowlist (``*`` and
the verb, to spell the remedy). With the variable UNSET, one refused wire
``reset`` logged two ``[safety]`` lines claiming the host pre-approves ``*``
and ``reset`` for every peer, and spent the memo, so the genuine warning for
those spellings never fired later in the process. It also broke the gate's
stated contract that the matcher is free of side effects.

Now the matcher is pure and ``remote_motion_refusal`` warns after an ADMITTED
command, reading the operator's real value, only when a bare entry (and no
scoped one) is what admitted it.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from strands_robots import _motion_grants
from strands_robots.mesh import core as mesh_core

RESET = {"action": "reset"}
SAFETY_CHANNEL = "strands_robots.mesh.core"


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv(mesh_core.WIRE_MOTION_ALLOW_ENV, raising=False)
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()
    with mesh_core._bare_allow_warned_lock:
        mesh_core._bare_allow_warned.clear()
    yield
    with mesh_core._bare_allow_warned_lock:
        mesh_core._bare_allow_warned.clear()


def _safety_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "[safety]" in r.getMessage()]


class TestTheMatcherIsPure:
    @pytest.mark.parametrize("probe", [frozenset({"*"}), frozenset({"reset"}), frozenset({"reset@leader-1"})])
    def test_probing_the_matcher_logs_nothing_and_remembers_nothing(
        self, probe: frozenset[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger=SAFETY_CHANNEL):
            assert mesh_core.allow_match(probe, "reset", "leader-1") is True
            assert mesh_core.allow_match(probe, "reset", "leader-1") is True

        assert caplog.records == []
        assert mesh_core._bare_allow_warned == set()

    def test_the_scoped_spellings_match_one_actor_and_the_bare_ones_every_actor(self) -> None:
        assert mesh_core.allow_match(frozenset({"reset@leader-1"}), "reset", "leader-1") is True
        assert mesh_core.allow_match(frozenset({"reset@leader-1"}), "reset", "attacker") is False
        assert mesh_core.allow_match(frozenset({"reset@leader-1"}), "reset", None) is False
        assert mesh_core.allow_match(frozenset({"*@leader-1"}), "step", "leader-1") is True
        assert mesh_core.allow_match(frozenset({"reset"}), "reset", None) is True
        assert mesh_core.allow_match(frozenset({"*"}), "reset", "anyone") is True
        assert mesh_core.allow_match(frozenset({"step"}), "reset", "leader-1") is False


class TestTheWarningFollowsTheDecision:
    def test_a_refusal_with_the_variable_unset_says_nothing_on_the_safety_channel(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger=SAFETY_CHANNEL):
            refused = mesh_core.remote_motion_refusal("reset", "so101", RESET, actor="attacker")

        assert refused is not None
        assert "needs operator approval" in refused[0]
        assert _safety_records(caplog) == []
        assert mesh_core._bare_allow_warned == set()

    def test_a_refusal_with_a_scoped_entry_for_someone_else_says_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(mesh_core.WIRE_MOTION_ALLOW_ENV, "reset@leader-1")

        with caplog.at_level(logging.DEBUG, logger=SAFETY_CHANNEL):
            refused = mesh_core.remote_motion_refusal("reset", "so101", RESET, actor="attacker")

        assert refused is not None
        assert _safety_records(caplog) == []
        assert mesh_core._bare_allow_warned == set()

    def test_a_bare_entry_that_admits_the_command_warns_exactly_once_across_two_calls(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(mesh_core.WIRE_MOTION_ALLOW_ENV, "reset")

        with caplog.at_level(logging.WARNING, logger=SAFETY_CHANNEL):
            first = mesh_core.remote_motion_refusal("reset", "so101", RESET, actor="leader-1")
            second = mesh_core.remote_motion_refusal("reset", "so101", RESET, actor="attacker")

        assert first is None and second is None
        warned = _safety_records(caplog)
        assert len(warned) == 1
        assert "names reset" in warned[0] and "reset@<peer>" in warned[0]
        assert mesh_core._bare_allow_warned == {"reset"}

    def test_a_scoped_entry_that_admits_the_command_warns_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(mesh_core.WIRE_MOTION_ALLOW_ENV, "reset@leader-1")

        with caplog.at_level(logging.DEBUG, logger=SAFETY_CHANNEL):
            assert mesh_core.remote_motion_refusal("reset", "so101", RESET, actor="leader-1") is None

        assert _safety_records(caplog) == []
        assert mesh_core._bare_allow_warned == set()

    def test_a_scoped_entry_beside_a_bare_one_is_the_quiet_path_for_its_actor(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The scoped spelling is what admitted leader-1; the bare one is reported only when it is what decided."""
        monkeypatch.setenv(mesh_core.WIRE_MOTION_ALLOW_ENV, "reset@leader-1,step")

        with caplog.at_level(logging.WARNING, logger=SAFETY_CHANNEL):
            assert mesh_core.remote_motion_refusal("reset", "so101", RESET, actor="leader-1") is None
            assert _safety_records(caplog) == []
            assert (
                mesh_core.remote_motion_refusal("step", "so101", {"action": "step", "steps": 1}, actor="leader-1")
                is None
            )

        assert [m for m in _safety_records(caplog) if "names step" in m]
        assert mesh_core._bare_allow_warned == {"step"}

    def test_a_grant_or_the_bypass_flag_admitting_the_command_does_not_blame_the_allowlist(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        with caplog.at_level(logging.WARNING, logger=SAFETY_CHANNEL):
            assert mesh_core.remote_motion_refusal("reset", "so101", RESET, actor="leader-1") is None

        assert _safety_records(caplog) == []
        assert mesh_core._bare_allow_warned == set()
