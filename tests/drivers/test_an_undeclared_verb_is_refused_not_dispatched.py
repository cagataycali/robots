"""An agent verb a driver's schema does not declare is refused, not dispatched.

A driver's ``tool_spec`` carries an ``action`` ``enum``. That enum *describes*
the verbs an agent may send; nothing between the model and
:meth:`~strands_robots.drivers.base.HardwareDriver.stream` enforces it. So a
dispatcher whose last branch is a bare ``else`` runs its final verb for every
value the enum does not cover - and on every shipped native driver that final
verb is the *write* one, the halt.

``FeetechDriver`` already refuses, and
``tests/drivers/test_feetech_driver.py::test_an_undeclared_verb_is_refused_not_run``
states why in the past tense: "The fallthrough branch used to be ``stop``, so
any verb outside the schema released torque on every motor and answered
``success``. ``home`` is the one an agent reaches for first - this driver
deliberately does not declare it - and a held payload dropped while the
transcript read as a clean move." That reasoning was never applied to the other
eleven drivers, so on those a typo (``"sensor"``), a verb borrowed from a sibling
(``"sensors"`` on a ``URDriver``, whose read verb is spelled ``state``) or a
non-string all halted the robot and reported ``status="success"``.

The population here is **derived from the registry** rather than listed, so a
driver added later is graded on arrival - the same relation shape
``test_agent_stop_verb_reports_the_halt_outcome.py`` uses for its own fleet
rule. Two things are graded, because the envelope alone is not the safety
property: the refusal must *say* the verb is unknown, and the halt must not
have *happened*.

The vocabulary divergence the fleet already carries is left as it is and only
made visible: ``CrazyflieDriver`` declares ``land`` where its siblings declare
``stop``, and ``URDriver`` declares ``state`` where its siblings declare
``sensors``. Refusing and naming the declared verbs lets an agent correct itself
in the same turn; silently running the halt for both spellings is what this
module refuses.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from typing import Any

import pytest

import strands_robots.drivers as drivers_pkg
from strands_robots.drivers.base import declared_verbs, undeclared_verb_error
from strands_robots.drivers.registry import get_native_driver_class

#: Every driver class shipped today. A walk that silently found fewer than this
#: would make every parametrized cell below vacuous.
_MINIMUM_DRIVER_CLASSES = 12

#: Verbs no shipped schema declares. ``None`` is included because the schema
#: says ``"type": "string"`` and nothing enforces that either, and ``"sensor"``
#: because a one-character typo is the case an agent actually produces.
_UNDECLARED = ("home", "halt", "SENSORS", "sensor", "", None)

#: Attributes a fallthrough could have reached. Named so the halt cell can
#: refuse to pass on a driver whose halt it never managed to observe.
_HALT_MEMBERS = ("stop_task", "stop")


def _driver_classes() -> dict[str, type[Any]]:
    """Every distinct native driver class the registry can build, by name."""
    return {
        cls.__name__: cls
        for cls in (get_native_driver_class(robot) for robot in drivers_pkg.list_native_drivers())
        if cls is not None
    }


def _first_robot_for(cls: type[Any]) -> str:
    """A registry name that resolves to ``cls`` or to a base of it.

    Read through the MRO so a subclass built inside a test - the widened-schema
    driver below - inherits the name its base is registered under, rather than
    needing a registry entry of its own.
    """
    for candidate in cls.__mro__:
        for robot in drivers_pkg.list_native_drivers():
            if get_native_driver_class(robot) is candidate:
                return robot
    raise AssertionError(f"{cls.__name__} is in the population but no robot names it")


def _build(cls: type[Any]) -> Any:
    """Build ``cls`` off hardware, through the factory's own contract."""
    return cls(tool_name=_first_robot_for(cls), cameras=None, data_config=None)


def _invoke(driver: Any, action: Any) -> dict[str, Any]:
    """Drive one agent invocation and return its single envelope."""

    async def _drive() -> dict[str, Any]:
        results = [
            event
            async for event in driver.stream({"toolUseId": "call-1", "name": "t", "input": {"action": action}}, {})
        ]
        assert len(results) == 1, f"the verb must yield exactly one result, got {len(results)}"
        return results[0]

    return asyncio.run(_drive())


def _text(envelope: dict[str, Any]) -> str:
    """Join every text block, for substring assertions."""
    return " ".join(str(block.get("text", "")) for block in envelope.get("content") or [] if isinstance(block, dict))


_CLASSES = sorted(_driver_classes().items())


def test_the_population_is_the_whole_fleet() -> None:
    """Without this, a registry walk that found nothing would grade nothing."""
    found = _driver_classes()
    assert len(found) >= _MINIMUM_DRIVER_CLASSES, (
        f"expected at least {_MINIMUM_DRIVER_CLASSES} native driver classes, found {sorted(found)}"
    )


@pytest.mark.parametrize(("name", "cls"), _CLASSES, ids=[n for n, _ in _CLASSES])
@pytest.mark.parametrize("action", _UNDECLARED, ids=[repr(a) for a in _UNDECLARED])
def test_an_undeclared_verb_is_refused_and_names_the_declared_ones(name: str, cls: type[Any], action: Any) -> None:
    """The refusal has to be actionable: a caller cannot guess the vocabulary."""
    driver = _build(cls)
    verbs = declared_verbs(driver.tool_spec)
    assert action not in verbs, f"premise: {action!r} must be outside {name}'s schema"

    envelope = _invoke(driver, action)

    assert envelope["status"] == "error", f"{name} ran undeclared verb {action!r} instead of refusing it"
    text = _text(envelope)
    for verb in verbs:
        assert verb in text, f"{name}'s refusal must name declared verb {verb!r}: {text}"


@pytest.mark.parametrize(("name", "cls"), _CLASSES, ids=[n for n, _ in _CLASSES])
@pytest.mark.parametrize("action", _UNDECLARED, ids=[repr(a) for a in _UNDECLARED])
def test_an_undeclared_verb_does_not_reach_the_halt(name: str, cls: type[Any], action: Any) -> None:
    """The envelope is half of it; the robot must not have been halted.

    A refusal that arrived *after* the halt ran would satisfy the cell above and
    still be the defect - the fallthrough's whole cost is that a read request
    stopped the robot.
    """
    driver = _build(cls)
    called: list[str] = []
    patched = [member for member in _HALT_MEMBERS if hasattr(driver, member)]
    assert patched, f"{name} exposes none of {_HALT_MEMBERS}, so no halt could be observed"
    for member in patched:
        # The stand-in answers a success envelope rather than ``None``, and
        # matches the sync/async shape of the member it replaces, so a
        # dispatcher that reaches the halt fails on this cell's own assertion
        # rather than on a ``TypeError`` from awaiting or unpacking the result.
        def _record(*args: Any, _member: str = member, **kwargs: Any) -> dict[str, Any]:
            called.append(_member)
            return {"status": "success", "content": [{"text": f"{_member} stand-in"}]}

        async def _record_async(*args: Any, _member: str = member, **kwargs: Any) -> dict[str, Any]:
            return _record(_member=_member)

        original = getattr(driver, member)
        setattr(driver, member, _record_async if inspect.iscoroutinefunction(original) else _record)

    _invoke(driver, action)

    assert called == [], f"{name} halted the robot for undeclared verb {action!r} (called {called})"


@pytest.mark.parametrize(("name", "cls"), _CLASSES, ids=[n for n, _ in _CLASSES])
def test_every_declared_verb_is_dispatched(name: str, cls: type[Any]) -> None:
    """The control, and the other half of the contract.

    A verb in the schema must have a branch: a declared verb the driver refuses
    as unknown is worse than one it never declared. Only the *unknown-verb*
    refusal is refused here - a declared verb may still fail on "not connected",
    which is the honest answer off hardware.
    """
    driver = _build(cls)
    for verb in declared_verbs(driver.tool_spec):
        text = _text(_invoke(driver, verb))
        assert "unknown action" not in text, f"{name} declares {verb!r} but its dispatcher does not handle it: {text}"


@pytest.mark.parametrize(("name", "cls"), _CLASSES, ids=[n for n, _ in _CLASSES])
def test_the_refusal_reads_its_verb_list_off_the_schema(name: str, cls: type[Any]) -> None:
    """A restated list drifts the first time a schema gains or loses a verb.

    A driver that widens its own schema is the case that catches it: the extra
    verb has to appear in the refusal without any edit to the refusal.
    """

    class _WiderSchema(cls):  # type: ignore[valid-type,misc]
        """A driver whose schema declares one verb more than its base."""

        @property
        def tool_spec(self) -> Any:
            spec = super().tool_spec
            spec["inputSchema"]["json"]["properties"]["action"]["enum"].append("wiggle")
            return spec

    driver = _build(_WiderSchema)
    assert "wiggle" in declared_verbs(driver.tool_spec)
    text = _text(_invoke(driver, "home"))
    assert "wiggle" in text, f"{name} restated a verb list instead of reading its schema: {text}"


@pytest.mark.parametrize(("name", "cls"), _CLASSES, ids=[n for n, _ in _CLASSES])
def test_no_dispatcher_ends_in_a_bare_else(name: str, cls: type[Any]) -> None:
    """The structural half, so the next driver is held to this on arrival.

    Read as source rather than behaviour because a driver may grow a verb whose
    branch this module cannot drive off hardware, while the *shape* - a terminal
    ``else`` that refuses rather than acts - is checkable either way.
    """
    function = ast.parse(textwrap.dedent(inspect.getsource(cls.stream))).body[0]
    assert isinstance(function, ast.AsyncFunctionDef | ast.FunctionDef)
    chains = [
        node
        for node in function.body
        if isinstance(node, ast.If) and "action" in ast.unparse(node.test) and node.orelse
    ]
    assert len(chains) == 1, f"{name}.stream has {len(chains)} action dispatch chains, expected exactly 1"
    # Follow the elif chain to its own tail; a nested if/else inside one branch
    # is a different decision and is not what this grades.
    node = chains[0]
    while len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
        node = node.orelse[0]
    body = "".join(ast.unparse(statement) for statement in node.orelse)
    assert undeclared_verb_error.__name__ in body, (
        f"{name}.stream's terminal else runs a verb instead of refusing an undeclared one: {body}"
    )
