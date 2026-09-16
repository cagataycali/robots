"""The headless refusal names ``<allow_env>=<value>``, and says when the variable is set to something that pre-approves nothing.

Before, a script with no operator was told ``Set STRANDS_ROBOT_COMMAND_ALLOW
or BYPASS_TOOL_CONSENT=true``. The natural readings - ``=1``, ``=true``,
``=yes`` - pre-approve nothing, because the variable takes the command's
own spelling (``execute``, ``/cmd_vel``, ``loco.SetVelocity``) or ``*``,
and the refusal came back unchanged with no hint that the value was the
problem. The value is asked of the tool's matcher (the same
``preapproval_setting`` the interrupt's ``how_to_answer`` line uses), so
the two never disagree.

A command is stopped with nobody to ask in two places, and only one of them
said anything: with no ``tool_context`` the refusal named the variable, and
when the host HAD a context but refused to interrupt, the refusal named the
reason and no remedy at all. Both now carry the same sentence from the same
helper. A command the operator DENIED is not one of those places - that
question was answered, so it keeps its bare refusal.

``=*`` is advertised only where it works. The ROS transports match through
``match_blocklist``, which canonicalises ``*`` to ``/*`` and matches nothing,
so a headless ``use_ros`` user who set ``STRANDS_ROS2_COMMAND_ALLOW=*`` on
the refusal's advice got the same refusal back, now saying the value named
"neither this command nor '*'" while offering ``=*`` again. Whether ``*`` is
offered is asked of the matcher, and the test asks the gate itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strands_robots.tools import _command_gate as gate_mod
from strands_robots.tools._command_gate import gate_motion, preapproval_setting
from tests.tools.test_command_gate_interrupt_carries_its_own_remedy import (  # noqa: F401
    IDS,
    SITES,
    TEST_ALLOW_ENV,
    _no_env,
)

pytestmark = pytest.mark.usefixtures("_no_env")


def _star_pre_approves(run, allow_env, monkeypatch) -> bool:
    """Whether ``<allow_env>=*`` really lets this site's command through - the only test of whether to offer it."""
    monkeypatch.setenv(allow_env, "*")
    try:
        return run(None) is None
    finally:
        monkeypatch.delenv(allow_env, raising=False)


def _star_clause(allow_env: str) -> str:
    return f"(or {allow_env}=* for every "


def _useless(allow_env: str, wrong: str, star: bool) -> str:
    names = "neither this command nor '*'" if star else "not this command"
    return f"{allow_env} is set to {wrong!r}, which names {names}, so it pre-approves nothing."


@pytest.mark.parametrize(("label", "run", "allow_env", "value"), SITES, ids=IDS)
class TestEverySiteNamesTheValue:
    def test_the_refusal_names_variable_equals_value(self, label, run, allow_env, value) -> None:
        text = str(run(None))
        assert "No tool_context available for operator approval" in text
        assert f"Set {allow_env}={value} " in text
        assert f"{gate_mod.BYPASS_CONSENT_ENV}=true" in text

    def test_star_is_offered_exactly_when_it_pre_approves(self, label, run, allow_env, value, monkeypatch) -> None:
        """Advice that loops back to the same refusal is worse than none: ``=*`` is named only where the matcher honours it."""
        offered = _star_clause(allow_env) in str(run(None))
        assert offered == _star_pre_approves(run, allow_env, monkeypatch)

    def test_the_named_value_really_pre_approves_the_call(self, label, run, allow_env, value, monkeypatch) -> None:
        monkeypatch.setenv(allow_env, value)
        assert run(None) is None

    @pytest.mark.parametrize("wrong", ["1", "true", "yes"])
    def test_a_set_but_useless_value_is_called_out(self, label, run, allow_env, value, wrong, monkeypatch) -> None:
        star = _star_pre_approves(run, allow_env, monkeypatch)
        monkeypatch.setenv(allow_env, wrong)
        text = str(run(None))
        assert _useless(allow_env, wrong, star) in text
        assert f"Set {allow_env}={value} " in text  # the remedy still follows

    def test_an_unset_variable_is_not_called_out(self, label, run, allow_env, value) -> None:
        assert "is set to" not in str(run(None))


class TestTheRefusalAndTheInterruptAgree:
    def test_same_setting_in_both(self, monkeypatch) -> None:
        match = gate_mod._allow_exact_or_star("so101")
        setting = preapproval_setting("execute", "so101", TEST_ALLOW_ENV, match)
        headless = gate_motion("robot", "execute", "so101", "it moves.", None, allow_env=TEST_ALLOW_ENV)
        assert setting == f"{TEST_ALLOW_ENV}=so101"
        assert f"Set {setting} " in str(headless)

    def test_a_typo_in_a_ros_allowlist_is_called_out(self, monkeypatch) -> None:
        monkeypatch.setenv(gate_mod.COMMAND_ALLOW_ENV, "/cmd_vell")
        text = str(gate_mod.gate_command("publish", "/cmd_vel", None, tool="use_ros"))
        assert "is set to '/cmd_vell', which names not this command" in text
        assert f"Set {gate_mod.COMMAND_ALLOW_ENV}=/cmd_vel " in text
        assert "=*" not in text  # match_blocklist canonicalises '*' to '/*', which matches nothing

    def test_a_ros_allowlist_set_to_star_is_not_told_to_set_star(self, monkeypatch) -> None:
        """The exact loop: follow the old advice, get the old refusal, plus the advice again."""
        monkeypatch.setenv(gate_mod.COMMAND_ALLOW_ENV, "*")
        text = str(gate_mod.gate_command("publish", "/cmd_vel", None, tool="use_ros"))
        assert "is set to '*', which names not this command, so it pre-approves nothing." in text
        assert f"Set {gate_mod.COMMAND_ALLOW_ENV}=/cmd_vel (comma-separated) or" in text
        assert "=* for every" not in text

    def test_a_tool_whose_matcher_honours_star_offers_it(self) -> None:
        text = str(gate_motion("robot", "execute", "so101", "it moves.", None, allow_env=TEST_ALLOW_ENV))
        assert f"(or {TEST_ALLOW_ENV}=* for every robot command; comma-separated)" in text


def _refuses_to_interrupt() -> MagicMock:
    """A host that supplies a tool_context but will not raise an interrupt through it."""
    ctx = MagicMock()
    ctx.interrupt.side_effect = RuntimeError("interrupts are not supported in this host")
    return ctx


@pytest.mark.parametrize(("label", "run", "allow_env", "value"), SITES, ids=IDS)
class TestAHostThatCannotInterruptGetsTheSameRemedy:
    def test_the_refusal_names_variable_equals_value(self, label, run, allow_env, value) -> None:
        text = str(run(_refuses_to_interrupt()))
        assert "interrupts are not available" in text
        assert f"Set {allow_env}={value} " in text
        assert f"{gate_mod.BYPASS_CONSENT_ENV}=true" in text

    def test_a_set_but_useless_value_is_called_out(self, label, run, allow_env, value, monkeypatch) -> None:
        star = _star_pre_approves(run, allow_env, monkeypatch)
        monkeypatch.setenv(allow_env, "1")
        assert _useless(allow_env, "1", star) in str(run(_refuses_to_interrupt()))

    def test_both_unreachable_operator_refusals_carry_the_same_remedy(self, label, run, allow_env, value) -> None:
        """One helper for both, so no operator to ask reads the same either way."""
        remedy = f"Set {allow_env}={value} ("
        assert remedy in str(run(None))
        assert remedy in str(run(_refuses_to_interrupt()))


class TestADeniedCommandIsNotADeadEnd:
    def test_an_operator_who_said_no_is_not_told_how_to_bypass(self) -> None:
        """The question was answered, so the remedy would be advice to overrule the answer."""
        ctx = MagicMock()
        ctx.interrupt.return_value = "n"
        refusal = gate_motion("robot", "execute", "so101", "it moves.", ctx, allow_env=TEST_ALLOW_ENV)
        assert refusal == "execute to 'so101' was declined by the operator."
