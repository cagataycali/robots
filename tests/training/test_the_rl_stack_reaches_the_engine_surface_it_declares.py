"""Every engine call the RL stack makes is one ``SimEngine`` publishes.

``SimEnv`` and the trainers hold their backend as ``engine`` and reach it by
name. Nothing checked those names against the seam, because every test stood a
duck-typed double in its place: a double declares its own signature, so a
keyword the seam renames, an argument it narrows, or a method it never had is
answered by the double and the whole RL suite stays green while no backend
would take the call.

This reads the call sites out of ``strands_robots/training/rl/`` and binds each
against :class:`~strands_robots.simulation.base.SimEngine`, then requires
:class:`~tests.training._engine_stand_in.EngineStandIn` - the one stand-in
those tests now share - to implement each reached method itself rather than
inherit the refusal it gives an unreached one.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

from strands_robots.simulation.base import SimEngine
from tests.training._engine_stand_in import EngineStandIn

_RL = Path(__file__).resolve().parents[2] / "strands_robots" / "training" / "rl"

#: The seam the RL stack is known to reach. A scan that finds fewer names than
#: this is a scan that stopped matching, not a stack that stopped calling.
_KNOWN = frozenset({"list_robots", "robot_action_keys", "get_observation", "reset", "send_action"})

#: Named by no call site, but reached through the inherited ``robot_action_keys``
#: default, which mirrors it - so the stand-in answers it rather than refusing.
_REACHED_BY_THE_DEFAULT = frozenset({"robot_joint_names"})

#: The published surface the RL stack does not reach, which the stand-in
#: therefore refuses rather than answers.
_REFUSING = frozenset(
    {
        "add_object",
        "add_robot",
        "create_world",
        "destroy",
        "get_state",
        "remove_object",
        "remove_robot",
        "render",
        "step",
    }
)


@dataclass(frozen=True)
class Reached:
    """One call the RL stack makes on its engine."""

    module: str
    line: int
    method: str
    positional: int
    keywords: tuple[str, ...]

    def __str__(self) -> str:
        args = [f"<{self.positional} positional>"] if self.positional else []
        return f"{self.module}:{self.line} engine.{self.method}({', '.join(args + [f'{k}=' for k in self.keywords])})"


def _is_engine(node: ast.expr) -> bool:
    """True for ``engine``, ``self.engine``, ``self.env.engine`` and the like."""
    if isinstance(node, ast.Name):
        return node.id == "engine"
    return isinstance(node, ast.Attribute) and node.attr == "engine"


def _calls() -> list[Reached]:
    """Every ``engine.<method>(...)`` call in the RL package, read from source."""
    found: list[Reached] = []
    for path in sorted(_RL.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not _is_engine(node.func.value):
                continue
            found.append(
                Reached(
                    module=path.name,
                    line=node.lineno,
                    method=node.func.attr,
                    positional=len(node.args),
                    keywords=tuple(kw.arg or "**" for kw in node.keywords),
                )
            )
    return found


_REACHED = _calls()


def test_the_scan_found_the_calls_it_grades() -> None:
    """A regex-shaped scan that matches nothing would pass every rule below."""
    assert {call.method for call in _REACHED} >= _KNOWN, f"only found {sorted({c.method for c in _REACHED})}"


@pytest.mark.parametrize("call", _REACHED, ids=str)
class TestEveryCallCouldReachABackend:
    """Each call site is graded against the seam, not against a double."""

    def test_the_method_is_one_the_seam_publishes(self, call: Reached) -> None:
        assert hasattr(SimEngine, call.method), f"no backend publishes {call.method}"

    def test_the_arguments_bind_to_the_published_signature(self, call: Reached) -> None:
        # ``bind`` is what the interpreter does at the call: a renamed keyword,
        # a dropped default or a new required parameter all fail here, where a
        # double that declares its own parameters would have accepted the call.
        signature = inspect.signature(getattr(SimEngine, call.method))
        args = [object()] * (call.positional + 1)  # +1 for self
        signature.bind(*args, **{name: object() for name in call.keywords})

    def test_the_shared_stand_in_answers_it(self, call: Reached) -> None:
        # Not merely present: implemented HERE. The unreached methods are
        # inherited-refusal stubs, so a newly reached one must be taught.
        assert call.method in vars(EngineStandIn), f"EngineStandIn does not implement {call.method}"


def test_the_unreached_surface_refuses_instead_of_answering() -> None:
    """The stand-in cannot quietly satisfy a call nobody taught it."""
    assert _REFUSING.isdisjoint({call.method for call in _REACHED}), "a reached method is listed as refusing"
    for name in sorted(_REFUSING):
        with pytest.raises(AssertionError, match=f"does not reach SimEngine.{name}"):
            getattr(EngineStandIn(), name)()


def test_every_published_method_is_either_answered_or_refused() -> None:
    """A new method on the seam is a decision, not a silent default."""
    accounted = {call.method for call in _REACHED} | _REACHED_BY_THE_DEFAULT | _REFUSING
    assert set(SimEngine.__abstractmethods__) <= accounted, (
        f"unaccounted for: {sorted(set(SimEngine.__abstractmethods__) - accounted)}"
    )
