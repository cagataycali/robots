"""Behavior tests for the accepted ``policy_provider`` domain of a hardware task.

``policy_provider`` names what the policy is built from, so it is the value the
other pre-flight checks in ``Robot`` read the registry *about*:
``_policy_port_error`` reads its ``requires`` entry to decide whether a port is
mandatory, ``_policy_requires_error`` reads it for the checkpoint keywords.
Both were written to hand an unresolvable name to ``create_policy``, which
raises from ``_get_policy`` - after ``_connect_robot`` has energized the arm,
and, on the agent tool, after the operator has approved the rollout. That is
the shape both of those checks exist to close, left open for the argument that
names what is being built.

These tests pin that a provider no policy can be resolved from is refused where
its siblings already are:

    - before ``_connect_robot``, so a misspelling does not spend the bring-up
      window that method's own comment describes as "a motors-bus handshake plus
      per-camera warmup - seconds on a real arm";
    - before the operator is asked, so an approval is never spent on a command
      that cannot run;
    - as a *provider* problem. ``TestAPortTheProviderDoesNotReadIsRefused``
      states the rule - "answering it here would report a port problem for a
      provider problem" - and it held only for a supplied port; a missing one
      was reported as "policy_port is required" for a provider that does not
      exist, the wrong reason and one whose remedy leads to the next;
    - and only for a name that really cannot resolve: the declared aliases and
      the auto-discovered modules must keep building, which is the over-reach
      direction that matters most.

The approval prompt is pinned alongside it, because it describes the same
argument: a policy built in this process was announced to the operator as a
server ``at localhost:None``.

No serial/USB hardware is touched: the driver is an in-memory fake and the
connect path is a recording stub.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

import strands_robots.hardware_robot as hw_mod
from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.registry.policies import (
    list_policy_aliases,
    list_policy_providers,
    policy_provider_resolves,
    provider_reads_a_port,
)
from tests.test_hardware_policy_port_domain import _text, hw  # noqa: F401 - fixture reuse

# The RPCs graded here run as an allowlisted operator: authorization fails
# closed and is graded in test_device_connect_hardening.py, not here.
pytestmark = pytest.mark.usefixtures("named_rpc_caller")

# Spellings no policy can be resolved from. The first is the realistic one - a
# transposed/duplicated letter in the default provider - and the rest cover the
# other ways a name arrives wrong.
UNRESOLVABLE: list[str] = ["grooot", "gr00t", "GRoot ", "mokc", "lerobot-local", "no_such_provider"]

# Names absent from ``list_policy_providers()`` that nevertheless build today:
# the declared aliases resolve through the registry's alias map, and these two
# modules resolve through ``import_policy_class``'s auto-discovery fallback.
AUTO_DISCOVERED: tuple[str, ...] = ("composite", "persistent")


def _gate_prompt(robot: Any, **tool_input: Any) -> str:
    """The sentence the operator is shown, taken from the headless refusal.

    With no agent there is no operator to ask, and ``gate_motion`` returns the
    warning it would have prompted with verbatim - so this reads the real
    prompt without standing up an interrupt.
    """
    message = robot._gate_motion("execute", tool_input, {"toolUseId": "t", "name": "arm", "input": {}}, {})
    assert message is not None, "the headless path must refuse and echo the prompt"
    return message


@pytest.fixture
def gateable(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A ``Robot`` that can build an approval prompt, with no operator reachable."""
    for var in ("BYPASS_TOOL_CONSENT", "STRANDS_ROBOT_COMMAND_ALLOW"):
        monkeypatch.delenv(var, raising=False)
    robot = HwRobot.__new__(HwRobot)
    robot.tool_name_str = "so101"
    robot._shutdown_event = threading.Event()
    # No operator grant is on deposit, so the gate asks rather than spending one.
    monkeypatch.setattr(hw_mod, "consume_grant", lambda tool, tool_input: False)
    return robot


class TestAnUnresolvableProviderIsRefusedBeforeTheArmIsTouched:
    """The provider is judged where the port already is: before connect, before the claim."""

    @pytest.mark.parametrize("provider", UNRESOLVABLE)
    def test_start_task_refuses_without_connecting(self, hw: Any, provider: str) -> None:  # noqa: F811
        result = hw.start_task("pick up the cube", policy_port=5555, policy_provider=provider)
        assert result["status"] == "error"
        assert hw.connects == [], "the arm was energized for a provider that cannot resolve"
        assert hw.robot.sent_actions == [], "the arm was commanded"

    @pytest.mark.parametrize("provider", UNRESOLVABLE)
    def test_execute_task_refuses_without_connecting(self, hw: Any, provider: str) -> None:  # noqa: F811
        result = hw._execute_task_sync("pick up the cube", policy_port=5555, policy_provider=provider)
        assert result["status"] == "error"
        assert hw.connects == [], "the arm was energized for a provider that cannot resolve"

    @pytest.mark.parametrize("provider", UNRESOLVABLE)
    def test_the_refusal_names_the_provider_and_what_would_resolve(self, hw: Any, provider: str) -> None:  # noqa: F811
        text = _text(hw.start_task("go", policy_provider=provider))
        assert provider in text, "the caller cannot correct a spelling the refusal does not quote"
        assert "groot" in text and "mock" in text, "the refusal must offer names that do resolve"

    def test_start_task_reports_it_instead_of_a_started_task(self, hw: Any) -> None:  # noqa: F811
        """``start_task`` answers before the executor, where nobody is left to tell."""
        result = hw.start_task("go", policy_provider="grooot")
        assert result["status"] == "error"
        assert "Task started" not in _text(result)


class TestAProviderProblemIsNotReportedAsAPortProblem:
    """The rule ``TestAPortTheProviderDoesNotReadIsRefused`` states, in both directions.

    That class pins it for a *supplied* port, which ``_policy_port_error``
    leaves to the provider. With the port *missing* there was no registry entry
    to read, so the same unknown name was refused as "policy_port is required":
    a port problem reported for a provider problem, and the identical sentence a
    correctly spelled ``groot`` gets - so the two were indistinguishable.
    """

    @pytest.mark.parametrize("entry", ["start_task", "_execute_task_sync"])
    def test_a_misspelling_with_no_port_is_named_as_unknown(self, hw: Any, entry: str) -> None:  # noqa: F811
        text = _text(getattr(hw, entry)("go", policy_provider="grooot"))
        assert "grooot" in text
        assert "policy_port is required" not in text, "a provider problem reported as a port problem"

    def test_a_misspelling_reads_differently_from_a_correct_name_missing_a_port(self, hw: Any) -> None:  # noqa: F811
        misspelled = _text(hw.start_task("go", policy_provider="grooot"))
        correct = _text(hw.start_task("go", policy_provider="groot"))
        assert misspelled != correct, "the operator cannot tell a typo from a missing port"
        assert "policy_port is required" in correct, "the port refusal keeps its own wording"

    def test_the_prescribed_remedy_no_longer_leads_to_another_wrong_reason(self, hw: Any) -> None:  # noqa: F811
        """Pre-fix, adding the port the message asked for got the call *past* the
        pre-flight entirely, to fail after the arm was energized."""
        result = hw.start_task("go", policy_provider="grooot", policy_port=5555)
        assert result["status"] == "error" and "grooot" in _text(result)
        assert hw.connects == []


class TestEveryProviderThatResolvesStillRuns:
    """The over-reach direction: refusing a working name is the expensive error."""

    @pytest.mark.parametrize("provider", sorted(list_policy_providers()))
    def test_a_registered_provider_is_not_refused(self, provider: str) -> None:
        assert HwRobot._policy_provider_error(provider, "start_task") is None

    @pytest.mark.parametrize("alias", sorted(list_policy_aliases()))
    def test_a_declared_alias_is_not_refused(self, alias: str) -> None:
        """Aliases are legal spellings that ``list_policy_providers()`` does not list."""
        assert alias not in list_policy_providers()
        assert HwRobot._policy_provider_error(alias, "start_task") is None

    @pytest.mark.parametrize("provider", AUTO_DISCOVERED)
    def test_an_auto_discovered_module_is_not_refused(self, provider: str) -> None:
        """``import_policy_class`` resolves these with no registry entry to read."""
        from strands_robots.policies.factory import import_policy_class

        assert import_policy_class(provider) is not None
        assert HwRobot._policy_provider_error(provider, "start_task") is None

    def test_a_nameless_provider_is_left_to_the_checks_that_own_it(self) -> None:
        """Falsy is the "not named" spelling; the port check already reports it."""
        for empty in (None, ""):
            assert HwRobot._policy_provider_error(empty, "start_task") is None

    def test_a_pre_built_policy_makes_the_provider_inert(self, hw: Any) -> None:  # noqa: F811
        """``_execute_task_sync`` never resolves a provider it was handed a policy for."""
        from tests.test_hardware_policy_port_domain import _Policy

        result = hw._execute_task_sync(
            "go", policy_provider="grooot", policy_object=_Policy(), duration=0.01, n_steps=1
        )
        assert result["status"] != "error" or "grooot" not in _text(result)


class TestTheApprovalPromptDescribesThePolicyTruthfully:
    """What the operator reads is what the command does.

    ``at {host}:{port}`` was unconditional, so the 9 registered providers that
    declare no port - the in-process ones - were announced as a server at
    ``localhost:None``, an endpoint that does not exist.
    """

    def test_an_in_process_policy_is_not_announced_as_a_server(self, gateable: Any) -> None:
        prompt = _gate_prompt(gateable, instruction="wave", policy_provider="mock")
        assert "localhost:None" not in prompt
        assert "policy mock built in this process, no server" in prompt

    def test_a_supplied_port_is_still_named(self, gateable: Any) -> None:
        prompt = _gate_prompt(gateable, instruction="wave", policy_provider="groot", policy_port=5555)
        assert "policy groot at localhost:5555" in prompt

    def test_a_server_dialing_provider_that_defaults_its_port_is_not_called_serverless(self, gateable: Any) -> None:
        """``cosmos3`` dials a server while defaulting the port, so "no server"
        would be a new false statement rather than a fix."""
        prompt = _gate_prompt(gateable, instruction="wave", policy_provider="cosmos3")
        assert "localhost:None" not in prompt
        assert "no server" not in prompt
        assert "default port" in prompt

    @pytest.mark.parametrize("provider", sorted(list_policy_providers()))
    def test_no_provider_is_ever_announced_at_a_none_port(self, gateable: Any, provider: str) -> None:
        assert "None" not in _gate_prompt(gateable, instruction="wave", policy_provider=provider)

    def test_the_prompt_still_names_the_robot_the_action_and_the_instruction(self, gateable: Any) -> None:
        """The over-reach control: the rest of the sentence is unchanged."""
        prompt = _gate_prompt(gateable, instruction="wave", policy_provider="mock")
        assert "'execute'" in prompt and "'so101'" in prompt and "'wave'" in prompt
        assert "operator approval" in prompt


class TestAnUnresolvableProviderNeverReachesTheOperator:
    """The approval is the expensive step, so the refusal comes first."""

    def test_the_pre_gate_check_refuses_it(self) -> None:
        robot = HwRobot.__new__(HwRobot)
        robot._shutdown_event = threading.Event()
        for action, method in (("execute", "execute_task"), ("start", "start_task")):
            err = robot._pre_gate_error(action, None, "grooot", 30.0)
            assert err is not None and "grooot" in _text(err)
            assert err["content"][0]["text"].startswith(f"{method}:")

    def test_a_budget_is_still_judged_first(self) -> None:
        """Order control: the duration refusal predates this check and keeps its place."""
        robot = HwRobot.__new__(HwRobot)
        robot._shutdown_event = threading.Event()
        err = robot._pre_gate_error("execute", None, "grooot", 0)
        assert err is not None and "duration" in _text(err)

    def test_a_sound_command_still_reaches_the_operator(self, gateable: Any) -> None:
        """The over-reach control: a resolvable provider is still gated, not refused."""
        robot = HwRobot.__new__(HwRobot)
        robot._shutdown_event = threading.Event()
        assert robot._pre_gate_error("execute", None, "mock", 30.0) is None
        assert "operator approval" in _gate_prompt(gateable, instruction="wave", policy_provider="mock")


class TestTheResolutionPredicateMatchesTheImporterItSpeaksFor:
    """``policy_provider_resolves`` must agree with ``import_policy_class``."""

    @pytest.mark.parametrize(
        "provider", sorted(set(list_policy_providers()) | set(list_policy_aliases()) | set(AUTO_DISCOVERED))
    )
    def test_every_name_the_importer_resolves_is_reported_as_resolving(self, provider: str) -> None:
        from strands_robots.policies.factory import import_policy_class

        try:
            imported = import_policy_class(provider) is not None
        except ImportError:
            # The provider exists but an optional dependency does not. Reporting
            # that as an unknown name would send the caller to check a spelling
            # that was correct, so it must still resolve.
            imported = True
        assert imported is policy_provider_resolves(provider) is True

    @pytest.mark.parametrize("provider", UNRESOLVABLE)
    def test_a_name_the_importer_refuses_is_reported_as_unresolvable(self, provider: str) -> None:
        from strands_robots.policies.factory import import_policy_class

        with pytest.raises(ValueError, match="Unknown policy provider"):
            import_policy_class(provider)
        assert policy_provider_resolves(provider) is False

    def test_the_describer_and_the_registry_agree_on_who_dials_a_server(self) -> None:
        """The prompt's three cases are the registry's answer, not a second opinion."""
        for provider in list_policy_providers():
            described = HwRobot._policy_description(provider, "localhost", None)
            if provider_reads_a_port(provider):
                assert "default port" in described
            else:
                assert "no server" in described


class TestTheGateAcceptsEverySpellingCreatePolicyAccepts:
    """The pre-flight refusal must not be stricter than ``create_policy`` itself.

    ``create_policy`` resolves in three stages - the runtime registry the public
    ``register_policy()`` API fills, smart strings, then the shipped registry -
    and a gate that mirrored only the last refused a provider the user had just
    registered, at every hardware entry point, while ``create_policy`` built it.
    """

    @pytest.fixture
    def registered(self, monkeypatch: pytest.MonkeyPatch) -> str:
        from strands_robots.policies import factory
        from strands_robots.policies.mock import MockPolicy

        monkeypatch.setattr(factory, "_runtime_registry", dict(factory._runtime_registry))
        monkeypatch.setattr(factory, "_runtime_aliases", dict(factory._runtime_aliases))
        factory.register_policy("gate_probe_custom", lambda: MockPolicy, aliases=["gate_probe"])
        return "gate_probe_custom"

    def test_a_runtime_registered_provider_is_not_refused(self, registered: str) -> None:
        from strands_robots.policies.factory import create_policy, provider_can_be_created

        assert type(create_policy(registered)).__name__ == "MockPolicy"
        assert provider_can_be_created(registered) is True
        assert HwRobot._policy_provider_error(registered, "start_task") is None

    def test_a_runtime_alias_is_not_refused(self, registered: str) -> None:
        assert HwRobot._policy_provider_error("gate_probe", "start_task") is None

    @pytest.mark.parametrize("smart", ["lerobot/act_so101_test", "zmq://127.0.0.1:5555", "ws://host:8000/policy"])
    def test_a_smart_string_is_left_to_resolution(self, smart: str) -> None:
        assert HwRobot._policy_provider_error(smart, "start_task") is None

    def test_the_refusal_lists_what_create_policy_would_accept(self, registered: str) -> None:
        err = HwRobot._policy_provider_error("grooot", "start_task")
        assert err is not None
        assert registered in _text(err)

    def test_an_unknown_name_is_still_refused_after_a_registration(self, registered: str) -> None:
        from strands_robots.policies.factory import provider_can_be_created

        assert provider_can_be_created("grooot") is False
        assert provider_can_be_created(None) is False
        assert provider_can_be_created(3) is False  # type: ignore[arg-type]
