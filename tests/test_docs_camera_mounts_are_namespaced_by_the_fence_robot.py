"""A documented ``add_camera(parent_body="<ns>/<body>")`` must use a robot the page built.

``Robot("unitree_g1")`` namespaces every body of the arm it builds under the
name it was given (``unitree_g1/torso_link``); ``Robot("g1")`` names the same
body ``g1/torso_link``. ``add_camera`` refuses a ``parent_body`` it cannot find
with ``status=error``, so a fence that mounts on ``g1/torso_link`` while the
page's sim is ``Robot("unitree_g1")`` never adds the camera the prose describes.

This grades every ``python`` fence in ``docs/**/*.md`` and ``README.md``: a
literal ``parent_body`` of the form ``<ns>/<body>`` must have ``<ns>`` bound by
a literal sim-mode ``Robot("<ns>")`` (or ``name="<ns>"``) somewhere on the same
page. A page that builds no robot in a fence (a snippet page) is not graded.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import strands_robots

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _documentation_files() -> list[Path]:
    files = sorted((_REPO_ROOT / "docs").rglob("*.md"))
    readme = _REPO_ROOT / "README.md"
    return [*files, readme] if readme.is_file() else files


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _literal(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None  # a fence with ``...`` placeholders is prose, not code


def _sim_robot_names(fences: list[str]) -> set[str]:
    """Names a page's sim-mode ``Robot(...)`` calls register their robot under."""
    names: set[str] = set()
    for fence in fences:
        tree = _parse(fence)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _callee_name(node) == "Robot"):
                continue
            mode = _keyword(node, "mode")
            if mode is not None and _literal(mode) != "sim":
                continue
            name = _literal(_keyword(node, "name")) or (_literal(node.args[0]) if node.args else None)
            if name:
                names.add(name)
    return names


def _namespaced_mounts(source: str) -> list[str]:
    """Literal ``parent_body`` values of the form ``<ns>/<body>`` in one fence."""
    tree = _parse(source)
    if tree is None:
        return []
    mounts: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _callee_name(node) == "add_camera":
            mount = _literal(_keyword(node, "parent_body"))
            if mount and "/" in mount:
                mounts.append(mount)
    return mounts


def _unbound_mounts(fences: list[str]) -> list[str]:
    names = _sim_robot_names(fences)
    if not names:
        return []
    return [m for fence in fences for m in _namespaced_mounts(fence) if m.split("/", 1)[0] not in names]


def test_every_documented_camera_mount_names_a_robot_the_page_built() -> None:
    offenders: list[str] = []
    for path in _documentation_files():
        text = path.read_text(encoding="utf-8")
        fences = [m.group(1) for m in _PYTHON_FENCE.finditer(text)]
        names = _sim_robot_names(fences)
        for mount in _unbound_mounts(fences):
            offenders.append(f"{path.relative_to(_REPO_ROOT)} mounts on {mount!r}; the page builds {sorted(names)}")
    assert not offenders, (
        "add_camera refuses a parent_body it cannot find (status=error), and bodies are "
        "namespaced by the name passed to Robot(...):\n  " + "\n  ".join(offenders)
    )


def test_the_grader_sees_a_mount_outside_the_page_namespace() -> None:
    page = ['sim = Robot("unitree_g1")\n', 'sim.add_camera(name="head", parent_body="g1/torso_link")\n']
    assert _unbound_mounts(page) == ["g1/torso_link"]
    assert _unbound_mounts(['sim = Robot("g1")\n', 'sim.add_camera(name="head", parent_body="g1/torso_link")\n']) == []
    assert _unbound_mounts(['sim.add_camera(name="wrist", parent_body="so101/gripper")\n']) == []
