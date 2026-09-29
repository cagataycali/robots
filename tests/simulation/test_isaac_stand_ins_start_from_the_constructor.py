"""No test restates ``IsaacSimulation.__init__`` on a ``__new__`` skeleton.

A skeleton that copies the constructor's defaults field by field is a second,
silently stale copy of ``__init__``: a field the constructor gains is missing
from every copy, and the method under test then raises ``AttributeError`` for a
reason unrelated to what the test is about. ``tests.simulation._isaac_engine``
starts from the real constructor instead. A bare ``__new__`` stays legal where
partial construction IS the subject (the finalizer and ``repr`` tests), which is
why this grades the restated default, not the ``__new__`` call.
"""

from __future__ import annotations

import ast
import queue
import threading
from pathlib import Path

import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from tests.simulation._isaac_engine import isaac_engine

_TESTS = Path(__file__).resolve().parents[1]
_SKELETON = "IsaacSimulation.__new__(IsaacSimulation)"
_DEFAULTS = vars(isaac_engine())


def _restates_a_default(attr: str, value: str) -> bool:
    if attr not in _DEFAULTS or attr == "_init_complete":
        return False
    try:
        given = eval(value, {"threading": threading, "queue": queue, "IsaacConfig": IsaacConfig})  # noqa: S307 - test-tree literals
    except Exception:
        return False  # names a local: the test models something, it does not restate
    default = _DEFAULTS[attr]
    if attr == "_main_tid":
        return value.strip() == "threading.get_ident()"
    if isinstance(default, queue.Queue) or type(default) is type(threading.RLock()):
        return type(given) is type(default)
    return type(given) is type(default) and given == default


def restated_defaults(source: str) -> list[str]:
    """``engine.<attr>`` assignments right after a skeleton that restate ``__init__``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        body = getattr(node, "body", None)
        for i, stmt in enumerate(body if isinstance(body, list) else []):
            if not (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)):
                continue
            if ast.get_source_segment(source, stmt.value) != _SKELETON:
                continue
            for follow in body[i + 1 :]:
                target = follow.targets[0] if isinstance(follow, ast.Assign) else None
                if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)):
                    break
                if target.value.id != stmt.targets[0].id:
                    break
                if _restates_a_default(target.attr, ast.get_source_segment(source, follow.value) or ""):
                    found.append(f"{follow.lineno}: {target.attr}")
    return found


@pytest.mark.parametrize(
    ("line", "flagged"),
    [
        ("e._objects = {}", True),
        ("e._lock = threading.RLock()", True),
        ("e._config = IsaacConfig()", True),
        ("e._objects = {'cube': 1}", False),
        ("e._world = stub_world", False),
        ("e._init_complete = False", False),
    ],
)
def test_the_grader_tells_a_restated_default_from_a_modelled_value(line: str, flagged: bool) -> None:
    source = f"def build():\n    e = {_SKELETON}\n    {line}\n"
    assert bool(restated_defaults(source)) is flagged


def test_the_engine_carries_every_constructor_field_with_its_finalizer_off() -> None:
    engine = isaac_engine()
    assert set(vars(engine)) == set(vars(IsaacSimulation()))
    assert engine._init_complete is False


def test_no_test_restates_the_constructor_on_a_skeleton() -> None:
    offenders = {
        str(path.relative_to(_TESTS)): hits
        for path in sorted(_TESTS.rglob("*.py"))
        if _SKELETON in (text := path.read_text(encoding="utf-8")) and (hits := restated_defaults(text))
    }
    assert offenders == {}, "build from tests.simulation._isaac_engine.isaac_engine instead"
