"""A robot's agent verbs live on its driver, not behind a handle no model can send.

A ``@tool`` function's input schema is what the model plans against, and every
parameter in it must be something the model can *emit* - JSON. A parameter
annotated :class:`~typing.Any` and required is not: the six rover verbs that
used to front :class:`~strands_robots.drivers.earthrover.EarthRoverDriver` each
took ``driver: Any``, which reached the schema as ``{"driver": {}}`` - required,
typed with nothing - and every value a model can put there is refused by the
verb's own handle check::

    rover_move(driver="earthrover", linear=0.3)
    -> "rover_move: `driver` of type 'str' does not expose a twist write (`move`)"
    rover_move(linear=0.3)
    -> "Validation failed for input parameters: driver Field required"

So the verb was unreachable from an agent in both directions, while the driver's
own ``tool_spec`` declared only reads and a halt - a rover an agent could watch
and not drive. The remedy is not a better refusal but the seam that needs no
handle: a driver *is* the agent's tool, its ``stream`` already holds the handle
as ``self``, and its ``tool_spec`` enum is where a robot's verbs belong. The
same relation the mesh, the teleop rail and every ``Robot(mode="real")`` caller
already uses.

Two things are graded. The rover's capabilities are reachable through the verbs
the driver declares - the spelling may change, the capability may not - and the
package-wide population of handle-taking verbs is an exact recorded set, so a
new family cannot be added silently. The families still in that set are the ones
this consolidation has not reached yet; each is a driver of its own with the
same seam available.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import strands_robots
import strands_robots.tools
from strands_robots.drivers.base import declared_verbs
from strands_robots.drivers.earthrover import EarthRoverDriver

_PACKAGE_ROOT = pathlib.Path(strands_robots.__file__).resolve().parent

#: The verb each ``rover_*`` tool became. Keyed by the name that is gone, so
#: this table also grades that it *is* gone: a consolidation that left both
#: doors open would be two names for one room.
ROVER_VERB_MOVED_TO: dict[str, str] = {
    "rover_move": "move",
    "rover_stop": "stop",
    "rover_lamp": "lamp",
    "rover_state": "sensors",
    "rover_camera": "camera",
    "rover_speak": "speak",
}

#: Every ``@tool`` that still requires a live handle, by module, with the number
#: of verbs it carries. Recorded rather than tolerated: these are robot families
#: whose verbs belong on their own driver's ``tool_spec`` the way the rover's now
#: are, and until they move, a *new* handle-taking verb fails here instead of
#: shipping. ``run_policy`` is the same shape one layer up - it requires a live
#: ``Simulation``.
HANDLE_TAKING_VERBS: dict[str, int] = {
    "tools/g1/g1_actions.py": 13,
    "tools/g1/g1_battery.py": 1,
    "tools/g1/g1_imu.py": 1,
    "tools/g1/g1_lidar_state.py": 1,
    "tools/g1/g1_lidar_summary.py": 1,
    "tools/g1/g1_mainboard.py": 1,
    "tools/g1/g1_pressure.py": 1,
    "tools/g1/g1_run_policy.py": 1,
    "tools/g1/g1_send_action.py": 1,
    "tools/g1/g1_start_task.py": 1,
    "tools/g1/g1_state.py": 1,
    "tools/g1/g1_stop_task.py": 1,
    "tools/g1/g1_task_status.py": 1,
    "tools/reachy/reachy_actions.py": 12,
    "tools/reachy/reachy_reads.py": 2,
    "tools/run_policy.py": 1,
}

#: A walk that resolved somewhere else, or stopped finding tools, would make the
#: population assertion pass by finding nothing.
MINIMUM_AGENT_TOOLS = 60


def _is_agent_tool(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``func`` carries ``@tool`` in either the bare or configured form."""
    for decorator in func.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "tool":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "tool":
            return True
    return False


def _required_any_parameters(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The parameters ``func`` requires that are annotated ``Any``.

    Required *and* ``Any`` is the pair that makes a verb unreachable: a
    parameter with a default is one the model can leave out, and an annotation
    the schema can render is one the model can fill in. ``Any`` renders as no
    type at all, which is how a live Python object arrives in a schema.
    """
    positional = func.args.posonlyargs + func.args.args
    required = positional[: len(positional) - len(func.args.defaults)]
    required += [
        arg for arg, default in zip(func.args.kwonlyargs, func.args.kw_defaults, strict=True) if default is None
    ]
    return [arg.arg for arg in required if arg.annotation is not None and ast.unparse(arg.annotation) == "Any"]


def _scan() -> tuple[int, dict[str, int]]:
    """Return the number of agent tools found and the handle-taking ones by module."""
    total = 0
    handle_taking: dict[str, int] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) or not _is_agent_tool(node):
                continue
            total += 1
            if _required_any_parameters(node):
                key = str(path.relative_to(_PACKAGE_ROOT))
                handle_taking[key] = handle_taking.get(key, 0) + 1
    return total, handle_taking


_TOTAL, _HANDLE_TAKING = _scan()


def test_the_scan_reached_the_package() -> None:
    """Without this the population assertion could pass on an empty walk."""
    assert _TOTAL >= MINIMUM_AGENT_TOOLS, f"found only {_TOTAL} agent tools under {_PACKAGE_ROOT}"


def test_no_new_verb_hides_behind_a_handle() -> None:
    """The handle-taking population is exactly the families not yet consolidated."""
    assert _HANDLE_TAKING == HANDLE_TAKING_VERBS


def test_the_rover_verbs_take_no_handle() -> None:
    """The family this guard was added for carries none."""
    assert [module for module in _HANDLE_TAKING if "earthrover" in module] == []


@pytest.mark.parametrize("gone", sorted(ROVER_VERB_MOVED_TO))
def test_the_tool_shell_is_gone_rather_than_kept_beside_the_driver(gone: str) -> None:
    """One capability, one door: the old name resolves nowhere."""
    assert not hasattr(strands_robots, gone)
    assert not hasattr(strands_robots.tools, gone)
    assert gone not in strands_robots.__all__
    assert gone not in strands_robots.tools.__all__


@pytest.mark.parametrize(("gone", "verb"), sorted(ROVER_VERB_MOVED_TO.items()))
def test_every_capability_is_reachable_as_a_declared_verb(gone: str, verb: str) -> None:
    """Deleting the shell must not delete the capability it fronted."""
    assert verb in declared_verbs(EarthRoverDriver(tool_name="earthrover").tool_spec), gone
