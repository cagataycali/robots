"""The fleet examples select a MuJoCo GL backend that exists on the host.

Measured at 6a8a8ea23 on macOS: every fleet example set
``os.environ.setdefault("MUJOCO_GL", "egl")`` at module scope, so the
documented live commands (``python examples/fleet/04_emergency_evacuation.py``)
died at ``import mujoco`` in under a second with ``RuntimeError: invalid value
for environment variable MUJOCO_GL: egl`` -- ``egl`` is Linux-only. The
tree's form, which the MUJOCO_GL linter recommends and the guarded examples
already use, picks ``cgl`` on macOS and ``egl`` elsewhere; an operator's own
``MUJOCO_GL`` still wins.
"""

from __future__ import annotations

import ast
import platform
from pathlib import Path

import pytest

from strands_robots.simulation.mujoco.backend import _mujoco_gl_valid_values
from tests.test_examples_mujoco_gl import _is_guarded_expr, _module_scope_gl_defaults

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FLEET_DIR = _REPO_ROOT / "examples" / "fleet"
_FLEET_EXAMPLES = sorted(_FLEET_DIR.glob("0*_*.py"))


def _module_scope_default_values(source: str) -> list[str]:
    """Every string a module-scope MUJOCO_GL default could resolve to."""
    values: list[str] = []
    for _lineno, expr_src in _module_scope_gl_defaults(source):
        values.extend(
            node.value for node in ast.walk(ast.parse(expr_src, mode="eval")) if isinstance(node, ast.Constant)
        )
    return [v for v in values if isinstance(v, str)]


def test_fleet_examples_exist() -> None:
    assert len(_FLEET_EXAMPLES) == 5, [p.name for p in _FLEET_EXAMPLES]


@pytest.mark.parametrize("path", _FLEET_EXAMPLES, ids=lambda p: p.name)
def test_fleet_example_gl_default_is_platform_guarded(path: Path) -> None:
    """A module-scope default must not name a platform-only backend unguarded.

    ``egl`` is invalid on macOS and ``cgl`` is invalid everywhere else, so a
    bare default of either is a guaranteed RuntimeError on some host.
    """
    source = path.read_text(encoding="utf-8")
    defaults = _module_scope_gl_defaults(source)
    assert defaults, f"{path.name}: no module-scope MUJOCO_GL default found"
    for lineno, expr_src in defaults:
        assert _is_guarded_expr(expr_src), (
            f"{path.name}:{lineno} MUJOCO_GL default {expr_src!r} is not platform-guarded; "
            'use \'"cgl" if sys.platform == "darwin" else "egl"\''
        )


@pytest.mark.parametrize("path", _FLEET_EXAMPLES, ids=lambda p: p.name)
def test_fleet_example_gl_default_resolves_to_a_backend_this_host_accepts(path: Path) -> None:
    """Evaluate the guarded expression on this host: the pick must be a valid value here."""
    source = path.read_text(encoding="utf-8")
    valid = _mujoco_gl_valid_values(platform.system())
    for _lineno, expr_src in _module_scope_gl_defaults(source):
        chosen = eval(expr_src, {"sys": __import__("sys")})  # noqa: S307 - the example's own literal expression
        assert chosen in valid, (
            f"{path.name}: MUJOCO_GL default {chosen!r} is not valid on {platform.system()} ({sorted(valid)})"
        )
