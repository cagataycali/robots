"""Newton ``add_robot`` refuses unusable requests with no solver installed.

:meth:`~strands_robots.simulation.newton.simulation.NewtonSimEngine.add_robot`
decides five caller-input refusals -- a world-less call, a duplicate name, an
unknown ``source``, an unsupported ``keyframe``, and an asset it cannot resolve
-- before it touches Newton or Warp. Nothing in that stretch needs a solver, so
every one of them is answerable on an install carrying neither.

The refusals are otherwise pinned only in modules gated on ``newton`` and
``warp`` being importable (``test_pre_world_guards.py``, whose own docstring
says it exists "so a new public method that forgets the guard ... is caught",
and ``test_robot_descriptions.py``). Those skip on an install without the
solver, which is the install a caller-input refusal most needs to be right on:
the message is all such a caller gets. Three of this method's other refusals --
the shared entity-name and pose-vector domains -- are already pinned by
un-gated cross-backend modules, so this file finishes that job for the rest.

The engine is a ``__new__`` skeleton carrying only the four attributes the
pre-solver stretch reads, so the real bound method runs and no solver is
constructed.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

#: Registered robot whose sim asset resolves from the shipped registry.
_KNOWN_ROBOT = "so100"

#: A name absent from every registry, so no asset can be resolved for it.
_UNKNOWN_ROBOT = "definitely-not-a-registered-robot"


def _engine(*, world: bool = True, rebuild_fatal: bool = False) -> Any:
    """Build the smallest engine the pre-solver stretch of ``add_robot`` reads.

    Args:
        world: When ``False``, leave ``_world`` unset so the world-less guard is
            the refusal under test.
        rebuild_fatal: When ``True``, make ``_rebuild`` raise. Every refusal in
            this file must answer without reaching it, which is what shows the
            decisions precede the solver rather than merely surviving it.

    Returns:
        A ``NewtonSimEngine`` built through ``__new__``, so the production
        method runs unchanged while no Newton or Warp object is constructed.
    """
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = SimWorld(timestep=0.002, gravity=[0.0, 0.0, -9.81]) if world else None
    engine._lock = threading.RLock()
    engine._robot_joint_map = {}

    def _rebuild() -> None:
        if rebuild_fatal:
            raise AssertionError("add_robot reached the solver rebuild for a request it must refuse")

    engine._rebuild = _rebuild  # type: ignore[method-assign]
    return engine


def _text(result: dict[str, Any]) -> str:
    """Return the human-readable text of a tool result envelope."""
    for block in result.get("content", []):
        if "text" in block:
            return str(block["text"])
    raise AssertionError(f"result carried no text block: {result!r}")


def _with_existing(name: str, *, rebuild_fatal: bool = False) -> Any:
    """An engine whose world already registers a robot called ``name``."""
    engine = _engine(rebuild_fatal=rebuild_fatal)
    engine._world.robots[name] = SimRobot(name=name, urdf_path="ignored.xml")
    return engine


#: The five caller-input refusals. Each entry drives ``add_robot`` on a fresh
#: engine, taking ``fatal`` so one table serves both the message assertions and
#: the ordering assertion (where reaching the solver rebuild raises). Keyed by
#: scenario so a failure names the case rather than a parametrize index.
_REFUSALS: dict[str, tuple[Callable[[bool], dict[str, Any]], str]] = {
    "no world": (
        lambda fatal: _engine(world=False, rebuild_fatal=fatal).add_robot(_KNOWN_ROBOT),
        "create_world",
    ),
    "duplicate name": (
        lambda fatal: _with_existing("dup", rebuild_fatal=fatal).add_robot("dup"),
        "already exists",
    ),
    "unknown source": (
        lambda fatal: _engine(rebuild_fatal=fatal).add_robot(_KNOWN_ROBOT, source="bogus"),
        "Unknown source",
    ),
    "keyframe unsupported": (
        lambda fatal: _engine(rebuild_fatal=fatal).add_robot(_KNOWN_ROBOT, keyframe="home"),
        "not yet supported",
    ),
    "asset unresolvable": (
        lambda fatal: _engine(rebuild_fatal=fatal).add_robot(_UNKNOWN_ROBOT),
        "list_robots",
    ),
}


class TestAddRobotRefusesWithoutASolver:
    """Each refusal answers through the envelope, naming what the caller can do."""

    @pytest.mark.parametrize("scenario", sorted(_REFUSALS))
    def test_the_refusal_is_a_structured_error(self, scenario: str) -> None:
        """No refusal raises past ``add_robot``'s documented result contract."""
        call, _token = _REFUSALS[scenario]
        result = call(False)
        assert isinstance(result, dict), f"{scenario}: expected a tool-result dict, got {type(result)!r}"
        assert result.get("status") == "error", f"{scenario}: expected status=error, got {result!r}"

    @pytest.mark.parametrize("scenario", sorted(_REFUSALS))
    def test_the_refusal_names_something_the_caller_can_act_on(self, scenario: str) -> None:
        """The message carries the remedy token, not just a failure."""
        call, token = _REFUSALS[scenario]
        message = _text(call(False))
        assert token in message, f"{scenario}: message does not name {token!r}: {message!r}"

    def test_the_unknown_source_refusal_lists_the_accepted_values(self) -> None:
        """A caller who guessed the source is told which values exist."""
        message = _text(_engine().add_robot(_KNOWN_ROBOT, source="bogus"))
        assert "registry" in message, f"accepted values missing from {message!r}"
        assert "robot_descriptions" in message, f"accepted values missing from {message!r}"

    def test_the_refusals_stay_distinguishable(self) -> None:
        """Five causes report five messages, so none is a copy of another."""
        messages = {scenario: _text(call(False)) for scenario, (call, _t) in _REFUSALS.items()}
        assert len(set(messages.values())) == len(messages), f"two refusals share one message: {messages!r}"

    @pytest.mark.parametrize("scenario", sorted(_REFUSALS))
    def test_the_refusal_precedes_the_solver(self, scenario: str) -> None:
        """A refused request builds nothing: reaching the rebuild raises here."""
        call, _token = _REFUSALS[scenario]
        assert call(True).get("status") == "error", f"{scenario}: expected the refusal, not a rebuild"

    def test_a_refused_request_registers_no_robot(self) -> None:
        """The world is left as it was, so a corrected retry is not blocked."""
        engine = _engine(rebuild_fatal=True)
        assert engine.add_robot(_UNKNOWN_ROBOT).get("status") == "error"
        assert engine._world.robots == {}, f"a refused add left state behind: {engine._world.robots!r}"


class TestTheContractNeedsNoSolver:
    """The premise: reaching these refusals imports neither Newton nor Warp."""

    def test_importing_the_engine_pulls_in_no_solver(self) -> None:
        """A child interpreter proves the import is solver-free."""
        root = Path(__file__).resolve().parents[3]
        assert (root / "pyproject.toml").is_file(), f"repo root misresolved: {root}"
        probe = (
            "import sys;"
            "from strands_robots.simulation.newton.simulation import NewtonSimEngine as E;"
            "assert E is not None;"
            "print(sorted(m for m in ('newton', 'warp') if m in sys.modules))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
            timeout=180,
        )
        assert proc.stdout.strip() == "[]", f"a solver was imported: {proc.stdout.strip()}"
