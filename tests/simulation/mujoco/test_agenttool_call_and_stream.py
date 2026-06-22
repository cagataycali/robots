"""AgentTool entry-point contract for the MuJoCo Simulation.

A ``Simulation`` is both a plain callable (``sim(action="render", ...)`` - the
form the README markets for ``Robot(...)``) and a Strands ``AgentTool`` whose
async ``stream()`` is what the agent runtime actually drives. Both wrap the same
``_dispatch_action`` routing but add their own boundary behavior:

* ``__call__`` rejects an empty/blank/non-string ``action`` with a friendly
  error dict (instead of raising ``TypeError`` or silently picking an action),
  then forwards keyword args as the action's parameters.
* ``stream()`` yields exactly one ``ToolResultEvent`` carrying the dispatch
  result tagged with the request's ``toolUseId``, and converts any unexpected
  exception into a standard error event rather than letting it escape the
  async generator.

These pin that boundary behavior without a GL context.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim():
    s = Simulation(tool_name="entrypoint_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


# __call__ - the plain-callable form.


def test_call_dispatches_action_to_method(sim):
    """sim(action="step", ...) routes through dispatch and runs the method."""
    res = sim(action="step", n_steps=2)
    assert res["status"] == "success"
    assert "2 steps" in res["content"][0]["text"]


def test_call_forwards_unknown_param_through_validation(sim):
    """Kwargs forwarded by __call__ still hit the validation layer."""
    res = sim(action="step", bogus_param=1)
    assert res["status"] == "error"
    assert "Unknown parameter 'bogus_param'" in res["content"][0]["text"]


@pytest.mark.parametrize("bad_action", ["", "   ", None, 123])
def test_call_blank_or_nonstring_action_errors_without_raising(sim, bad_action):
    """An empty/blank/non-string action returns an error dict, never raises."""
    res = sim(action=bad_action)
    assert res["status"] == "error"
    assert "requires action=" in res["content"][0]["text"]


def test_call_strips_surrounding_whitespace_from_action(sim):
    """A padded action name is trimmed before routing (not flagged unknown)."""
    res = sim(action="  get_state  ")
    assert res["status"] == "success"


# stream() - the AgentTool async path.


async def _collect(agen):
    return [event async for event in agen]


def test_stream_yields_single_result_tagged_with_tool_use_id(sim):
    """stream() emits one ToolResultEvent carrying the request's toolUseId."""
    import asyncio

    tool_use = {"toolUseId": "abc-123", "input": {"action": "get_state"}}
    events = asyncio.run(_collect(sim.stream(tool_use, {})))

    assert len(events) == 1
    result = events[0].tool_result
    assert result["status"] == "success"
    assert result["toolUseId"] == "abc-123"


def test_stream_converts_dispatch_exception_to_error_event(sim, monkeypatch):
    """An unexpected error inside dispatch becomes a standard error event,
    tagged with the toolUseId, instead of escaping the async generator."""
    import asyncio

    def _boom(*_args, **_kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(sim, "_dispatch_action", _boom)
    tool_use = {"toolUseId": "err-1", "input": {"action": "render"}}
    events = asyncio.run(_collect(sim.stream(tool_use, {})))

    assert len(events) == 1
    result = events[0].tool_result
    assert result["status"] == "error"
    assert result["toolUseId"] == "err-1"
    assert "Sim error: kaboom" in result["content"][0]["text"]
