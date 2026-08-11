"""The agent-facing entry points refuse an action the tool schema never advertised.

``_dispatch_action`` resolves by ``getattr`` with no allowlist, so every public
method of the engine is dispatchable while ``tool_spec.json``'s ``action`` enum
advertises a curated 77-entry subset. That width is a deliberate *Python*
convenience, but it used to reach a model too: an agent is handed the enum and
nothing else, so the 23 dispatchable-but-unadvertised capabilities - including
the ``TeleopMixin`` cluster, which drives real hardware from a host input
device - were invocable by name from a model that was never told they existed.

#2093 priced the two candidate refusal sites. This pins the one that was chosen:
the two agent-facing entry points (``__call__`` and ``stream``) refuse a
non-enum action, and ``_dispatch_action`` itself stays wide so the Python path
is unchanged. Both halves matter, and the second is the one a regression would
break silently - ``examples/`` reaches ``get_observation`` through the router
directly, and every Python-only capability has its own ``.method()`` callers.

The Python-only inventory is imported from ``test_tool_spec`` rather than
restated: that set is what the tool-spec guards maintain, and a second copy
would drift from it in exactly the direction that makes this suite pass while
the contract is broken.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import (  # noqa: E402
    _PUBLISHED_ACTIONS,
    _TOOL_SPEC_PATH,
    MuJoCoSimEngine,
    Simulation,
)
from tests.simulation.mujoco.test_tool_spec import _PYTHON_ONLY_ACTIONS  # noqa: E402

_NOT_FOR_AGENTS = "is not available to an agent"


@pytest.fixture
def sim():
    s = Simulation(tool_name="agent_boundary_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _text(result: dict) -> str:
    return result["content"][0]["text"]


def _is_agent_refusal(result: object) -> bool:
    """True when *result* is the unadvertised-action refusal.

    Deliberately total over any return value: the Python path hands back
    whatever the method returns, which for several of these is a plain float or
    a dataclass rather than a tool-result dict.
    """
    if not isinstance(result, dict) or result.get("status") != "error":
        return False
    first = (result.get("content") or [{}])[0]
    return _NOT_FOR_AGENTS in (first.get("text") or "")


def _stream_once(sim, action, **params) -> dict:
    """Drive one agent tool call through ``stream`` and return its result dict."""

    async def _run():
        tool_use = {"toolUseId": "tu-1", "input": {"action": action, **params}}
        return [event async for event in sim.stream(tool_use, {})]

    events = asyncio.run(_run())
    assert len(events) == 1
    return events[0].tool_result


# The published set is derived, not restated.


def test_published_actions_is_exactly_the_schema_enum():
    """The guard's allowlist is the advertised enum itself, with nothing added.

    A hand-maintained second list is the failure this is written against: it
    would let the tool refuse an action its own schema tells the model to use.
    """
    spec = json.loads(Path(_TOOL_SPEC_PATH).read_text())
    assert _PUBLISHED_ACTIONS == frozenset(spec["properties"]["action"]["enum"])


# __call__ - the plain-callable form the README markets.


@pytest.mark.parametrize("action", sorted(_PYTHON_ONLY_ACTIONS))
def test_call_refuses_every_deliberately_python_only_action(sim, action):
    """No member of the Python-only inventory is reachable as sim(action=...)."""
    result = sim(action=action)
    assert result["status"] == "error"
    assert _NOT_FOR_AGENTS in _text(result)


def test_call_refusal_names_the_action_and_points_at_the_schema(sim):
    """The refusal is actionable: it names the action and where the list lives."""
    text = _text(sim(action="teleoperate"))
    assert "teleoperate" in text
    assert "tool_spec" in text


def test_call_refuses_a_python_only_action_before_reading_its_parameters(sim):
    """The refusal precedes validation, so a valid call is refused on the action
    alone rather than drawing a parameter complaint that implies it would work."""
    result = sim(action="get_observation", robot_name="nope", skip_images=True)
    assert result["status"] == "error"
    assert _NOT_FOR_AGENTS in _text(result)
    assert "Unknown parameter" not in _text(result)


def test_call_reports_a_nonexistent_action_as_unknown_not_as_python_only(sim):
    """A typo keeps the "Unknown action" verdict: it is a different problem from
    a capability held back on purpose, and conflating them sends a reader
    hunting for a misspelling that is not there."""
    result = sim(action="teleprot")
    assert result["status"] == "error"
    assert "Unknown action: teleprot" in _text(result)
    assert _NOT_FOR_AGENTS not in _text(result)


def test_call_reports_a_property_as_unknown_without_evaluating_it(sim):
    """A non-callable attribute is not a capability, so it reads as unknown.

    The guard probes the class rather than the instance for this reason: a
    property evaluated to word a refusal would run engine code on the error
    path, and ``tool_name`` is a property that exists on the type.
    """
    result = sim(action="tool_name")
    assert result["status"] == "error"
    assert "Unknown action: tool_name" in _text(result)


def test_call_still_dispatches_a_published_action(sim):
    """The positive control: the 77 advertised actions are unaffected."""
    result = sim(action="step", n_steps=2)
    assert result["status"] == "success"
    assert "2 steps" in _text(result)


def test_call_still_strips_whitespace_before_the_published_check(sim):
    """Trimming happens first, so a padded published action is not refused as
    unadvertised - the guard sees the same name dispatch would have seen."""
    assert sim(action="  get_state  ")["status"] == "success"


def test_call_blank_action_keeps_its_own_message(sim):
    """An empty action is answered by the entry point's own check, which is more
    specific than the unadvertised-action refusal."""
    result = sim(action="   ")
    assert result["status"] == "error"
    assert "requires action=" in _text(result)


# Aliases - the advertised spelling is what counts.


@pytest.mark.parametrize("alias", sorted(MuJoCoSimEngine._ACTION_ALIASES))
def test_an_action_alias_is_accepted_at_the_agent_boundary(sim, alias):
    """Aliases exist so a model's alternate spelling routes; refusing one would
    defeat that. Every alias is in the enum today, so this holds through the
    enum - it is written to catch an alias added outside it, which is the case
    where the guard would silently break the alias mechanism."""
    assert sim(action=alias)["status"] == "success"


def test_an_alias_target_method_name_is_not_itself_advertised(sim):
    """``list_robots`` is the published name; ``list_robots_info`` is the method
    it resolves to and is not advertised, so a model naming the internal
    spelling is refused. Deliberate: the enum is the contract."""
    result = sim(action="list_robots_info")
    assert result["status"] == "error"
    assert _NOT_FOR_AGENTS in _text(result)


# stream() - the path the agent runtime actually drives.


@pytest.mark.parametrize("action", ["get_observation", "teleoperate", "save_episode"])
def test_stream_refuses_a_python_only_action(sim, action):
    """The refusal covers the async AgentTool path, not only the callable form.

    Guarding one entry point and not the other would make the advertised
    contract hold on the form a test reaches for and not on the form a model
    actually arrives through.
    """
    result = _stream_once(sim, action)
    assert result["status"] == "error"
    assert _NOT_FOR_AGENTS in _text(result)


def test_stream_refusal_carries_the_tool_use_id(sim):
    """A refused call is still a well-formed tool result, so the runtime can
    match it to the request rather than dropping it."""
    result = _stream_once(sim, "teleoperate")
    assert result["toolUseId"] == "tu-1"


def test_stream_reports_a_nonexistent_action_as_unknown(sim):
    result = _stream_once(sim, "definitely_not_an_action")
    assert result["status"] == "error"
    assert "Unknown action: definitely_not_an_action" in _text(result)


def test_stream_still_dispatches_a_published_action(sim):
    result = _stream_once(sim, "get_state")
    assert result["status"] == "success"


# The Python path is deliberately untouched.


@pytest.mark.parametrize("action", sorted(_PYTHON_ONLY_ACTIONS - {"cleanup", "teleoperate", "stop_teleoperate"}))
def test_the_router_still_reaches_every_python_only_action(sim, action):
    """``_dispatch_action`` keeps its full width: the refusal lives at the two
    agent entry points, not in the router.

    This is the half a regression would break invisibly - ``examples/`` calls
    ``sim._dispatch_action("get_observation", ...)`` directly. The assertion is
    only that the action is *reached*, not that it succeeds: most need a robot,
    a recording session or a GL context, and their own suites cover their
    behaviour. ``cleanup`` and the teleop start/stop pair are excluded because
    reaching them tears down or blocks the fixture rather than returning a
    verdict.

    Reaching the method is what is asserted, in the two shapes that can prove
    it. Several of these return a native value - ``physics_timestep`` a float,
    ``get_camera_params`` a dataclass, ``describe`` a dict with no ``content``
    key - and that alone shows the Python path is unmediated. Others raise
    (``get_frame`` needs OpenGL). Either is past the guard, because the guard
    returns a dict and never raises.
    """
    try:
        result = sim._dispatch_action(action, {})
    except Exception:
        return
    assert not _is_agent_refusal(result)


def test_the_router_still_refuses_a_leading_underscore_action(sim):
    """Unchanged: a private name was never dispatchable and still is not."""
    result = sim._dispatch_action("_compile_world", {"action": "_compile_world"})
    assert result["status"] == "error"
    assert "Unknown action" in _text(result)
