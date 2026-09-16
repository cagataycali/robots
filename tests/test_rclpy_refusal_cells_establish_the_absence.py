# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cell expecting an rclpy refusal must make rclpy absent, not assume it is.

``rclpy`` ships with a ROS 2 distribution rather than from PyPI, so on the hosts
this suite usually runs on it is simply missing and ``require_optional("rclpy")``
raises without being asked to. A cell can then asssert an ``ImportError`` and
pass while establishing nothing - and on a host with a distro sourced the same
cell reaches the real transport instead.

That is not a hypothetical. :class:`~strands_robots.ros_telemetry.RosTelemetryBridge`
continues past its probe into ``rclpy.init()`` and ``create_node()``, so on a
sourced host four cells here constructed a live node on the domain they were
checking the *guard* for, left ``rclpy.ok()`` true for the rest of the session,
and destroyed neither. Measured with ROS 2 Jazzy sourced: ``rclpy.ok()`` False
before, True after, node ``strands_robots`` on domain 11 still alive.

``sys.meta_path`` is not the way to establish the absence either. An import
consults ``sys.modules`` first and only reaches the finders when it misses, so a
finder refusing ``rclpy`` is bypassed once anything in the session has imported
it: one cell carried such a finder, documented as holding "whether or not the
interpreter running the suite happens to have a ROS 2 distro sourced", and
failed in the full suite while passing alone.

:func:`tests._blocked_module.blocked` does both halves - ``sys.modules[name] =
None`` and dropping ``require_optional``'s memo - and restores both, so it holds
wherever it runs. This grader reports a cell that expects the refusal without it.

The rule is derived from the tree rather than listed: the surfaces are the
classes and the functions whose own source calls ``require_optional("rclpy",
...)``, so a new rclpy-probing bridge or guard is covered the day it lands.

Both shapes are needed, because a cell reaches the probe two ways. Constructing
a bridge is one. Calling the function that carries the probe is the other, and
it is the cheaper one to write: ``Robot._check_ros2_bridge_deps`` raises the
refusal a ``Robot(..., ros2_bridge=True)`` would, with no lerobot install and no
device, so a cell grading that message calls it directly. Covering only the
classes left that shape unguarded, and a cell took it: it read the host with
``importlib.util.find_spec("rclpy")`` and skipped when the module was there,
which is the "assume the absence" this file exists to forbid wearing a guard.
Measured, that cell skipped on a host with Jazzy sourced and failed in the full
suite on a host without it -- ``tests/policies/moveit2/test_zmq_sidecar.py``
leaves its fake ``rclpy`` in ``require_optional``'s memo, so the probe answered
from there and raised nothing. Neither run graded the message.
"""

from __future__ import annotations

import ast
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PACKAGE = _REPO_ROOT / "strands_robots"
_TESTS = _REPO_ROOT / "tests"

#: What counts as establishing the absence: the shared helper, or replacing the
#: probe itself (a double that raises is as deterministic as a blocked import).
_ESTABLISHES = ("blocked", "require_optional")


def _probes_rclpy(node: ast.AST) -> bool:
    """True when this subtree calls ``require_optional("rclpy", ...)``."""
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "require_optional"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "rclpy"
        for call in ast.walk(node)
    )


def _rclpy_probing_classes() -> set[str]:
    """Classes whose *construction* probes for ``rclpy``, and their subclasses.

    Scoped to ``__init__`` because that is what a cell reaches by constructing
    the class. A class that probes rclpy from some other method only does so on
    the path that calls it - :class:`~strands_robots.hardware_robot.Robot` probes
    from ``_check_ros2_bridge_deps``, reached only for ``ros2_bridge=True``, so
    constructing a ``Robot`` for an unrelated missing dependency is not this rule's
    business. Subclasses are followed by base name: a bridge that probes in its
    base's ``__init__`` carries the same obligation without repeating the call.
    """
    classes: dict[str, list[str]] = {}
    probing: set[str] = set()
    for path in sorted(_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            classes[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]
            if any(
                isinstance(body, ast.FunctionDef) and body.name == "__init__" and _probes_rclpy(body)
                for body in node.body
            ):
                probing.add(node.name)
    # Transitive closure over the base names collected above.
    grew = True
    while grew:
        grew = False
        for name, bases in classes.items():
            if name not in probing and probing.intersection(bases):
                probing.add(name)
                grew = True
    return probing


def _rclpy_probing_functions() -> set[str]:
    """Function and method names whose own body calls ``require_optional("rclpy")``.

    A cell that calls one of these reaches the probe without constructing
    anything, so it carries the same obligation as a cell that builds a bridge.

    ``__init__`` is deliberately left to :func:`_rclpy_probing_classes`, which
    resolves it to the class it belongs to. Admitted here as a bare name it would
    match every cell that constructs any object at all, and report cells that go
    nowhere near rclpy.
    """
    return {
        node.name
        for path in sorted(_PACKAGE.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name != "__init__" and _probes_rclpy(node)
    }


def _surfaces() -> set[str]:
    """Every name a cell can reach the rclpy probe through."""
    return _rclpy_probing_classes() | _rclpy_probing_functions()


def _expects_an_import_error(item: ast.withitem) -> bool:
    """True when this ``with`` item is ``pytest.raises(ImportError, ...)``."""
    call = item.context_expr
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "raises"):
        return False
    return any(isinstance(a, ast.Name) and a.id == "ImportError" for a in call.args)


def _offending_cells(surfaces: set[str]) -> list[str]:
    """Report every test function expecting an rclpy refusal it does not establish."""
    offenders: list[str] = []
    for path in sorted(_TESTS.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef) or not func.name.startswith("test_"):
                continue
            names = {n.id for n in ast.walk(func) if isinstance(n, ast.Name)}
            names |= {n.attr for n in ast.walk(func) if isinstance(n, ast.Attribute)}
            # A patched probe names its target as a string: monkeypatch.setattr(
            # mod, "require_optional", double). That is establishing it too.
            names |= {n.value for n in ast.walk(func) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            reaches = names & surfaces
            expects = any(
                _expects_an_import_error(item) for w in ast.walk(func) if isinstance(w, ast.With) for item in w.items
            )
            if reaches and expects and not (names & set(_ESTABLISHES)):
                rel = path.relative_to(_REPO_ROOT)
                offenders.append(f"{rel}::{func.name} reaches {sorted(reaches)} expecting ImportError")
    return offenders


def test_the_probing_surfaces_are_discovered() -> None:
    """Non-vacuity: the derivation finds both shapes that really probe rclpy."""
    surfaces = _surfaces()

    assert {"RosTelemetryBridge", "HardwareRosBridge"} <= surfaces, surfaces
    assert "_check_ros2_bridge_deps" in surfaces, (
        "a guard that probes rclpy from a method is reachable without constructing "
        f"anything, so it is a surface too: {sorted(surfaces)}"
    )
    assert "__init__" not in surfaces, (
        "as a bare name __init__ matches every cell that constructs anything; the "
        "class rule resolves it to the class instead"
    )


def test_no_cell_assumes_rclpy_is_absent() -> None:
    offenders = _offending_cells(_surfaces())

    assert offenders == [], "these cells expect an rclpy refusal the host may not give:\n  " + "\n  ".join(offenders)
