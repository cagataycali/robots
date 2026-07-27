"""Every public class in the ``strands_robots.device_connect`` package must
document each of its public methods and properties.

The device_connect package is the surface that exposes a strands-robots
:class:`~strands_robots.robot.Robot`, a
:class:`~strands_robots.simulation.Simulation`, and a Pollen Reachy Mini head to
Device Connect's device registry as ``DeviceDriver`` adapters
(:class:`~strands_robots.device_connect.robot_driver.RobotDeviceDriver`,
:class:`~strands_robots.device_connect.sim_driver.SimulationDeviceDriver`,
:class:`~strands_robots.device_connect.reachy_mini_driver.ReachyMiniDriver`) plus
the real-time hardware links
(:class:`~strands_robots.device_connect.reachy_transport.ZenohLink` /
``WebSocketLink``). Several of these classes carried a rich class docstring but
left the ``identity`` / ``status`` accessors and the ``start`` / ``stop`` /
``send_cmd`` link methods undocumented, so a reader driving the driver blind
could not tell what an accessor returns or what a link method does without
reading its body. This mirrors the mesh guard
(``tests/mesh/test_public_member_docstrings.py``) and the policy guard
(``tests/policies/test_builtin_policy_docstrings.py``).

The check walks every module across the device_connect tree by AST (no import,
so it needs none of the optional extras - ``device_connect_edge``, ``zenoh``,
``websockets`` - installed) and fails if any public method or property of a
public class defines no docstring. Property setters and deleters are exempt: a
``@x.setter`` mirrors its documented getter and a docstring on it is noise,
matching the convention elsewhere in the tree.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots

_PACKAGE_DIR = Path(strands_robots.__file__).parent / "device_connect"


def _is_setter_or_deleter(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the function is a ``@<name>.setter`` or ``@<name>.deleter``."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr in ("setter", "deleter"):
            return True
    return False


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the function is a ``@typing.overload`` stub (body is documented once)."""
    for dec in node.decorator_list:
        name = dec.attr if isinstance(dec, ast.Attribute) else getattr(dec, "id", "")
        if name == "overload":
            return True
    return False


def _public_members_without_docstring(class_node: ast.ClassDef) -> list[str]:
    """Return names of public methods/properties in the class body lacking a docstring."""
    offenders: list[str] = []
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        if _is_setter_or_deleter(node) or _is_overload(node):
            continue
        if ast.get_docstring(node) is None:
            offenders.append(node.name)
    return offenders


def _public_classes() -> dict[str, ast.ClassDef]:
    """Map ``relpath::ClassName`` -> ClassDef for every public class in the tree."""
    classes: dict[str, ast.ClassDef] = {}
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        rel = path.relative_to(_PACKAGE_DIR)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{rel}::{node.name}"] = node
    return classes


def test_device_connect_dir_exists() -> None:
    """The device_connect package directory resolves and contains modules."""
    assert _PACKAGE_DIR.is_dir()
    assert any(_PACKAGE_DIR.rglob("*.py"))


def test_every_public_class_member_has_docstring() -> None:
    """No public method/property of a public device_connect class may lack a docstring."""
    missing: dict[str, list[str]] = {}
    for qualname, class_node in _public_classes().items():
        offenders = _public_members_without_docstring(class_node)
        if offenders:
            missing[qualname] = offenders

    assert not missing, "Public members missing docstrings:\n" + "\n".join(
        f"  {qualname}: {', '.join(names)}" for qualname, names in sorted(missing.items())
    )
