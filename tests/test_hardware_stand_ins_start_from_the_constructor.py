"""No test restates ``hardware_robot.Robot.__init__`` on a ``__new__`` skeleton.

A skeleton that copies the constructor's defaults field by field is a second,
silently stale copy of ``__init__``: a field the constructor gains is missing
from every copy, and the verb under test then raises ``AttributeError`` for a
reason unrelated to what the test is about. ``tests._hardware_robot`` starts from
the real constructor instead. A bare ``__new__`` stays legal where partial
construction IS the subject (the finalizer, ``_initialize_robot`` itself), which
is why this grades the restated default, not the ``__new__`` call.
"""

from __future__ import annotations

import ast
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from strands_robots.hardware_robot import Robot, RobotTaskState
from tests._daemon_executor import DaemonThreadExecutor
from tests._hardware_robot import hardware_robot_on

_TESTS = Path(__file__).resolve().parent
_MODULE = "strands_robots.hardware_robot"
_DEFAULTS = vars(hardware_robot_on())
_SCOPE = {
    "threading": threading,
    "RobotTaskState": RobotTaskState,
    "ThreadPoolExecutor": ThreadPoolExecutor,
    "DaemonThreadExecutor": DaemonThreadExecutor,
}
_EXECUTORS = (ThreadPoolExecutor, DaemonThreadExecutor)


def _restates_a_default(attr: str, value: str) -> bool:
    if attr not in _DEFAULTS or attr == "robot":
        return False
    try:
        given = eval(value, dict(_SCOPE))  # noqa: S307 - test-tree literals
    except Exception:
        return False  # names a local: the test models something, it does not restate
    default = _DEFAULTS[attr]
    if isinstance(default, _EXECUTORS):
        return isinstance(given, _EXECUTORS)
    if isinstance(default, (RobotTaskState, threading.Event)) or type(default) is type(threading.Lock()):
        return type(given) is type(default)
    return type(given) is type(default) and given == default


def _robot_names(tree: ast.Module) -> set[str]:
    """Spellings that name ``hardware_robot.Robot`` in one test module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == _MODULE:
            names |= {a.asname or a.name for a in node.names if a.name == "Robot"}
        elif isinstance(node, ast.ImportFrom) and node.module == "strands_robots":
            names |= {f"{a.asname or a.name}.Robot" for a in node.names if a.name == "hardware_robot"}
        elif isinstance(node, ast.Import):
            names |= {f"{a.asname}.Robot" for a in node.names if a.name == _MODULE and a.asname}
    return names


def restated_defaults(source: str) -> list[str]:
    """``hw.<attr>`` assignments right after a skeleton that restate ``__init__``."""
    tree = ast.parse(source)
    skeletons = {f"{name}.__new__({name})" for name in _robot_names(tree)}
    found = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for i, stmt in enumerate(body):
            if not (isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None):
                continue
            target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
            if not isinstance(target, ast.Name) or ast.unparse(stmt.value) not in skeletons:
                continue
            for follow in body[i + 1 :]:
                attr = follow.targets[0] if isinstance(follow, ast.Assign) else None
                if not (isinstance(attr, ast.Attribute) and isinstance(attr.value, ast.Name)):
                    break
                if attr.value.id != target.id:
                    break
                if _restates_a_default(attr.attr, ast.unparse(follow.value)):
                    found.append(f"{follow.lineno}: {attr.attr}")
    return found


@pytest.mark.parametrize(
    ("line", "flagged"),
    [
        ("hw.mesh = None", True),
        ("hw._task_claimed = False", True),
        ("hw._task_admission = threading.Lock()", True),
        ("hw._executor = ThreadPoolExecutor(max_workers=1)", True),
        ("hw.control_frequency = 50.0", True),
        ("hw.control_frequency = 500.0", False),
        ("hw.robot = None", False),
        ("hw.mesh = stub_mesh", False),
    ],
)
def test_the_grader_tells_a_restated_default_from_a_modelled_value(line: str, flagged: bool) -> None:
    source = f"from {_MODULE} import Robot as HwRobot\ndef build():\n    hw = HwRobot.__new__(HwRobot)\n    {line}\n"
    assert bool(restated_defaults(source)) is flagged


def test_the_robot_carries_every_constructor_field_and_the_stand_in() -> None:
    arm = object()
    hw = hardware_robot_on(arm, tool_name="so101", control_frequency=200.0)
    assert set(vars(hw)) == set(_DEFAULTS)
    assert (hw.robot, hw.tool_name_str, hw.action_sleep_time) == (arm, "so101", 1.0 / 200.0)
    assert isinstance(hw._executor, DaemonThreadExecutor)
    assert isinstance(hw, Robot)


def test_no_test_restates_the_constructor_on_a_skeleton() -> None:
    offenders = {
        str(path.relative_to(_TESTS)): hits
        for path in sorted(_TESTS.rglob("*.py"))
        if "__new__(" in (text := path.read_text(encoding="utf-8")) and (hits := restated_defaults(text))
    }
    assert offenders == {}, "build from tests._hardware_robot.hardware_robot_on instead"
