"""The driver error envelope is built in one place: ``drivers.base.refuse``.

Every driver verb that refuses answers ``{"status": "error", "content": [{"text": reason}]}``.
Thirteen drivers once carried a private copy of the one-line helper that builds it,
so a change to the envelope had thirteen places to miss. This grades the whole
``strands_robots.drivers`` tree for a function whose body is exactly that
return, under any name, and allows only the owner.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.drivers as drivers_pkg
from strands_robots.drivers.base import refuse

_DRIVERS = Path(drivers_pkg.__file__).parent


def _builds_the_envelope(fn: ast.FunctionDef) -> bool:
    """Is ``fn``'s last statement ``return {"status": "error", "content": [{"text": <its parameter>}]}``?"""
    if len(fn.args.args) != 1 or not fn.body or not isinstance(fn.body[-1], ast.Return):
        return False
    param = fn.args.args[0].arg
    shape = f'{{"status": "error", "content": [{{"text": {param}}}]}}'
    value = fn.body[-1].value
    return value is not None and ast.unparse(value) == ast.unparse(ast.parse(shape, mode="eval").body)


def _envelope_builders() -> list[str]:
    found = []
    for path in sorted(_DRIVERS.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and _builds_the_envelope(node):
                found.append(f"{path.relative_to(_DRIVERS)}:{node.name}")
    return found


def test_only_the_base_module_builds_the_refusal_envelope() -> None:
    assert _envelope_builders() == ["base.py:refuse"], (
        "a driver re-implemented the refusal envelope; import strands_robots.drivers.base.refuse instead"
    )


def test_the_envelope_is_the_documented_shape() -> None:
    assert refuse("stop: bus closed") == {"status": "error", "content": [{"text": "stop: bus closed"}]}
