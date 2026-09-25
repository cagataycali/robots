"""Sim examples must not pass a hardware-only embodiment to ``create_policy``.

A recurring documentation defect: a MuJoCo sim example is copied from a hardware
example and keeps ``embodiment="so_real"``. The ``*_real`` embodiments declare
the lerobot driver's ``<motor>.pos`` joint keys (e.g. ``shoulder_pan.pos``),
which never match the bare-numeric MuJoCo joint names (``"1".."6"``). The
``PackStateProcessorStep`` then finds zero state keys, never composes
``observation.state``, and a state-conditioned policy (MolmoAct2) fails deep
inside the lerobot processor pipeline with ``requires observation.state``.

This statically scans every example script (recursively across topic
subfolders): any example that constructs a
SIM robot (``Robot(...)`` without ``mode="real"``, or ``create_simulation(...)``)
and passes a hardware (``*_real``) embodiment to ``create_policy`` is a defect.
Pure-hardware examples (``Robot(..., mode="real")``) are exempt - ``so_real`` is
correct there.

A commented recipe counts as passing it. ``examples/02_policy_abstraction.py``
listed the spellings ``create_policy`` accepts as a block of comments, one of
them ``create_policy("allenai/MolmoAct2-SO100_101", embodiment="so_real")``,
above a ``Robot("so100")`` sim - and a reader copies that line as readily as the
code under it. Measured on the sim ``so100``: ``so_real`` declares
``shoulder_pan.pos .. gripper.pos``, of which the sim observation binds 0 of 6,
where the robot's own ``so100`` binds 6 of 6. Only a balanced, parseable
``create_policy(...)`` in a comment is read, so prose that merely names the
function is not graded.
"""

from __future__ import annotations

import ast
import io
import json
import tokenize
from pathlib import Path
from typing import TypeGuard

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_EMBODIMENTS_JSON = _REPO_ROOT / "strands_robots" / "policies" / "lerobot_local" / "embodiments.json"


def _hardware_embodiments() -> set[str]:
    """Names (configs + aliases) that resolve to a hardware (``*_real``) config.

    Derived from ``embodiments.json`` so the rule tracks the registry rather than
    a hand-maintained list. A config is hardware iff its name ends in ``_real``;
    an alias is hardware iff its target config is.
    """
    raw = json.loads(_EMBODIMENTS_JSON.read_text(encoding="utf-8"))
    configs = raw.get("configs", {})
    aliases = raw.get("aliases", {})
    hardware = {name for name in configs if name.endswith("_real")}
    hardware |= {alias for alias, target in aliases.items() if target in hardware}
    return hardware


def _string_value(node: ast.AST) -> str | None:
    """Return the literal string value of ``node``, else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_sim_robot_example(tree: ast.AST) -> bool:
    """True if the script constructs a sim robot.

    Sim = a ``Robot(...)`` call WITHOUT ``mode="real"`` (sim is the default), or a
    ``create_simulation(...)`` call. A script that only ever builds
    ``Robot(..., mode="real")`` robots is treated as hardware-only.
    """
    has_sim = False
    has_real_only_robot = False
    saw_robot = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == "create_simulation":
            return True
        if name == "Robot":
            saw_robot = True
            mode = None
            for kw in node.keywords:
                if kw.arg == "mode":
                    mode = _string_value(kw.value)
            if mode == "real":
                has_real_only_robot = True
            else:
                has_sim = True
    if has_sim:
        return True
    # Only real robots constructed -> not a sim example.
    if saw_robot and has_real_only_robot:
        return False
    return False


def _is_create_policy(node: ast.AST) -> TypeGuard[ast.Call]:
    """Whether ``node`` is a call to ``create_policy``, however it is imported."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)) == "create_policy"


def _commented_create_policy_calls(source: str) -> list[ast.Call]:
    """Every ``create_policy(...)`` call a COMMENT shows the reader.

    Parsed rather than pattern-matched: a fragment is read only when its
    parentheses balance and it parses as a call, so prose that names the
    function is left alone and a recipe a reader can copy is not.
    """
    calls: list[ast.Call] = []
    try:
        comments = [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        ]
    except tokenize.TokenError:
        return calls
    for comment in comments:
        start = 0
        while (opening := comment.find("create_policy(", start)) != -1:
            cursor = opening + len("create_policy(")
            depth = 1
            while cursor < len(comment) and depth:
                depth += {"(": 1, ")": -1}.get(comment[cursor], 0)
                cursor += 1
            start = cursor
            if depth:
                continue
            try:
                node = ast.parse(comment[opening:cursor], mode="eval").body
            except SyntaxError:
                continue
            if _is_create_policy(node):
                calls.append(node)
    return calls


def _create_policy_embodiments(source: str, tree: ast.AST) -> list[tuple[str, str]]:
    """``(embodiment, where)`` for every literal ``embodiment=`` ``create_policy`` is given.

    Both halves of the example's interface: the calls it makes and the recipes
    its comments hand the reader.
    """
    found: list[tuple[str, str]] = []
    for node, where in [(n, "code") for n in ast.walk(tree) if _is_create_policy(n)] + [
        (n, "a commented recipe") for n in _commented_create_policy_calls(source)
    ]:
        for kw in node.keywords:
            if kw.arg == "embodiment" and (val := _string_value(kw.value)) is not None:
                found.append((val, where))
    return found


def _example_scripts() -> list[Path]:
    if not _EXAMPLES_DIR.is_dir():
        return []
    return sorted(_EXAMPLES_DIR.rglob("*.py"))


@pytest.mark.parametrize("script", _example_scripts(), ids=[p.name for p in _example_scripts()])
def test_sim_example_uses_sim_embodiment(script: Path) -> None:
    """Sim examples must not pass a hardware (``*_real``) embodiment."""
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    if not _is_sim_robot_example(tree):
        pytest.skip(f"{script.name} is not a sim example")
    hardware = _hardware_embodiments()
    offending = [f"{name!r} in {where}" for name, where in _create_policy_embodiments(source, tree) if name in hardware]
    assert not offending, (
        f"{script.name} builds a sim robot but passes hardware embodiment(s) "
        f"{offending} to create_policy. Hardware ('*_real') embodiments declare "
        f"'<motor>.pos' joint keys that never match the MuJoCo bare-numeric joint "
        f"names; observation.state ends up empty. Use the sim embodiment (e.g. "
        f'"so101"/"so100") for MuJoCo.'
    )


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param('create_policy("m", embodiment="so_real")', ["so_real"], id="a-recipe-is-read"),
        pytest.param("see create_policy and pass an embodiment", [], id="prose-naming-the-function-is-not"),
        pytest.param('create_policy("m", embodiment="so100"', [], id="unbalanced-is-not-a-recipe"),
        pytest.param('create_policy("m", embodiment=)', [], id="unparseable-is-not-a-recipe"),
        pytest.param('create_policy("m", embodiment=NAME)', [], id="a-name-is-not-a-literal"),
    ],
)
def test_a_comment_is_read_only_where_it_shows_a_call(snippet: str, expected: list[str]) -> None:
    """The comment reader grades recipes a reader can copy, and nothing else.

    Without this the rule above passes on a tree that has no offender left,
    whatever the reader does.
    """
    source = f"# {snippet}\nx = 1\n"
    found = [name for name, where in _create_policy_embodiments(source, ast.parse(source)) if where != "code"]
    assert found == expected
