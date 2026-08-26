"""Teardown after a bring-up that failed in ``self.robot = ...`` reports nothing.

``Robot(..., mode="real")`` builds its lerobot device in one statement::

    self.robot = self._initialize_robot(robot, cameras, **kwargs)

When that statement raises -- an absent motor-SDK extra is the ordinary way, and
the message naming the extra is the whole value of that path -- the attribute it
assigns never comes into existence. ``cleanup()`` then runs from the factory's
rollback and from ``__del__``, reaches ``_disconnect_devices`` ->
``_close_open_devices``, and both of those read the device handle.

Reading it as ``getattr(self.robot, "bus", None)`` guards the *inner* attribute
while dereferencing ``self.robot`` itself, so teardown raised
``AttributeError: 'Robot' object has no attribute 'robot'``, which ``cleanup()``
caught and logged at ERROR. The operator saw a library-internal attribute name
beside the install hint they actually needed, and ``_close_open_devices`` -- the
method whose own docstring names "a failed connect" as one of its two callers --
could not survive the most partial failure there is.

The sibling teardown reader ``_shutdown_ros_bridge`` already spells this
correctly (``getattr(self, "_ros_bridge", None)``), so the fix is that spelling
in both device readers rather than a new convention.

Why every existing gate was silent: ``tests/test_robot_factory.py``'s
``test_mesh_attrs_set_before_initialize_robot_no_attribute_error_in_cleanup``
grades exactly this failure -- and selects offending records with
``"mesh" in r.message``, so a record naming any other attribute is outside
everything it reads. It passed while printing this defect on its own output line.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from strands_robots.hardware_robot import Robot as HwRobot

_LOGGER = "strands_robots.hardware_robot"

#: The statement whose failure this file is about. ``__init__`` assigns the
#: device handle from this call, so a raise here leaves the attribute absent.
_BRINGUP_CALL = "_initialize_robot"


def _bare_robot(tool_name: str = "probe") -> Any:
    """A ``Robot`` that never ran ``__init__``, as a failed bring-up leaves it.

    ``__new__`` rather than a constructed instance because the state under test
    is precisely the state a raised ``__init__`` leaves behind: no ``robot``
    attribute at all. ``tool_name_str`` is set because every teardown branch
    quotes it when it logs.

    Args:
        tool_name: Value for ``tool_name_str``.

    Returns:
        An uninitialised ``Robot`` instance.
    """
    robot = HwRobot.__new__(HwRobot)
    robot.tool_name_str = tool_name
    return robot


def _cleanup_errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every ``cleanup()`` error record captured, as messages.

    Args:
        caplog: pytest's log capture fixture.

    Returns:
        The messages of records ``cleanup()``'s own handler emitted.
    """
    return [r.message for r in caplog.records if "Cleanup error" in r.message]


def _class_methods() -> set[str]:
    """Names that resolve to a callable on the class, across the whole MRO.

    An attribute read like ``self.stop_teleoperate`` is a bound method from
    ``TeleopMixin``, not instance state, so a survey of instance attributes has
    to exclude it -- and excluding only names defined in ``Robot`` itself would
    keep every inherited method as a false positive.

    Returns:
        Every attribute name on the class that is callable.
    """
    return {name for name in dir(HwRobot) if callable(getattr(HwRobot, name, None))}


def _init_assignments() -> dict[str, int]:
    """First ``self.<attr> = ...`` line in ``__init__``, per attribute.

    Returns:
        Attribute name -> line number, numbered within ``__init__``'s own
        source so the result does not move when unrelated code shifts.
    """
    fn = ast.parse(textwrap.dedent(inspect.getsource(HwRobot.__init__))).body[0]
    first: dict[str, int] = {}
    for node in ast.walk(fn):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                first.setdefault(target.attr, node.lineno)
    return first


def _bringup_line() -> int:
    """Line in ``__init__`` of the assignment fed by the bring-up call.

    Returns:
        The line number, within ``__init__``'s own source, of the single
        statement that assigns from :data:`_BRINGUP_CALL`.
    """
    fn = ast.parse(textwrap.dedent(inspect.getsource(HwRobot.__init__))).body[0]
    found = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign) and _BRINGUP_CALL in ast.unparse(node.value)
    ]
    assert len(found) == 1, f"expected one {_BRINGUP_CALL} assignment in __init__, found {found}"
    return found[0]


def _teardown_closure() -> set[str]:
    """Methods reachable from ``cleanup()`` through ``self.<method>()`` calls.

    Returns:
        Method names, including ``cleanup`` itself.
    """
    seen: set[str] = set()
    pending = ["cleanup"]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        target = getattr(HwRobot, name, None)
        if not callable(target):
            continue
        try:
            source = textwrap.dedent(inspect.getsource(target))
        except (OSError, TypeError):
            continue
        seen.add(name)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                pending.append(node.func.attr)
    return seen


def _closure_established() -> set[str]:
    """Attributes the teardown closure itself establishes or probes for absence.

    Two spellings make a read safe without ``__init__`` having run:

    * a *lazy initializer* -- ``TeleopMixin._ensure_teleop_state`` assigns its
      whole family behind ``if not hasattr(self, "_teleops")``, so those
      attributes exist by the time anything in the closure reads them;
    * an explicit absence probe -- ``getattr(self, name, default)`` or
      ``hasattr(self, name)``, which is how ``_shutdown_ros_bridge`` and
      ``cleanup()``'s teleop branch read state that may not be there.

    Returns:
        Every attribute name the closure assigns or probes.
    """
    safe: set[str] = set()
    for name in sorted(_teardown_closure()):
        tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(HwRobot, name))))
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    safe.add(target.attr)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr")
                and len(node.args) > 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
                and isinstance(node.args[1], ast.Constant)
            ):
                safe.add(node.args[1].value)
    return safe


def _unprotected_state_reads() -> dict[str, str]:
    """Teardown reads of instance state nothing in the closure establishes.

    Returns:
        Attribute name -> the method that reads it.
    """
    methods = _class_methods()
    safe = _closure_established()
    out: dict[str, str] = {}
    for name in sorted(_teardown_closure()):
        tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(HwRobot, name))))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr not in methods
                and node.attr not in safe
            ):
                out.setdefault(node.attr, name)
    return out


class TestTheFailedBringUpIsTheStateUnderTest:
    """Premises: the shape of the failure, measured rather than assumed.

    Each of these holds on the unfixed code too -- they establish that the
    regression cells below are describing the real path.
    """

    def test_the_device_handle_is_assigned_by_the_statement_that_can_raise(self):
        """``self.robot`` cannot be pre-set: its assignment is the bring-up."""
        assignments = _init_assignments()
        assert "robot" in assignments, "__init__ no longer assigns self.robot"
        assert assignments["robot"] == _bringup_line(), (
            "self.robot is no longer assigned from the bring-up call, so the "
            "premise of this file (the attribute is absent when it raises) "
            "needs re-deriving"
        )

    def test_a_failed_bring_up_leaves_no_device_handle(self):
        """The attribute is genuinely absent, not merely ``None``."""
        with patch.object(HwRobot, _BRINGUP_CALL, side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                HwRobot(tool_name="probe", robot="so101_follower")
        assert not hasattr(_bare_robot(), "robot")

    def test_the_sibling_teardown_reader_guards_on_self(self):
        """``_shutdown_ros_bridge`` is the in-file precedent for the spelling."""
        source = inspect.getsource(HwRobot._shutdown_ros_bridge)
        assert 'getattr(self, "_ros_bridge", None)' in source

    def test_the_device_closer_documents_a_failed_connect_as_a_caller(self):
        """The method that raised is the one written for partial failure."""
        doc = HwRobot._close_open_devices.__doc__ or ""
        assert "failed connect" in doc.lower()


class TestTeardownAfterAFailedBringUpReportsNothing:
    """Regression: the defect, at each surface that exhibited it."""

    def test_cleanup_logs_no_error_when_the_bring_up_failed(self, caplog):
        """The whole symptom: an ERROR record beside the operator's real error."""
        with caplog.at_level("ERROR", logger=_LOGGER):
            with patch.object(HwRobot, _BRINGUP_CALL, side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError, match="boom"):
                    HwRobot(tool_name="probe", robot="so101_follower")
        assert _cleanup_errors(caplog) == []

    def test_the_error_named_the_missing_attribute_rather_than_the_cause(self, caplog):
        """No record may name the internal handle the operator did not ask about."""
        with caplog.at_level("ERROR", logger=_LOGGER):
            with patch.object(HwRobot, _BRINGUP_CALL, side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError, match="boom"):
                    HwRobot(tool_name="probe", robot="so101_follower")
        offenders = [m for m in _cleanup_errors(caplog) if "no attribute" in m]
        assert offenders == [], f"teardown reported a missing attribute: {offenders}"

    def test_closing_devices_tolerates_an_absent_device_handle(self):
        """``_close_open_devices`` is reached by the factory's own rollback."""
        _bare_robot()._close_open_devices()

    def test_disconnecting_devices_tolerates_an_absent_device_handle(self):
        """``_disconnect_devices`` is what ``cleanup()`` calls."""
        _bare_robot()._disconnect_devices()

    def test_cleanup_called_directly_on_a_failed_bring_up_reports_nothing(self, caplog):
        """``__del__`` reaches ``cleanup()`` on an object built this way."""
        robot = _bare_robot()
        robot._shutdown_event = SimpleNamespace(set=lambda: None)
        robot._task_state = SimpleNamespace(status=None)
        robot._executor = SimpleNamespace(shutdown=lambda wait: None)
        robot.mesh = None
        with caplog.at_level("ERROR", logger=_LOGGER):
            robot.cleanup()
        assert _cleanup_errors(caplog) == []


class TestAPresentDeviceHandleIsStillClosed:
    """Over-reach controls: tolerating an absent handle must not skip a present one.

    Every expectation here is one the unfixed code also met, so a failure in
    this class means the fix stopped closing a real device.
    """

    def test_a_connected_driver_is_disconnected_through_its_own_method(self):
        """The preferred path: the driver's ``disconnect()`` releases torque."""
        calls: list[str] = []
        robot = _bare_robot()
        robot.robot = SimpleNamespace(is_connected=True, disconnect=lambda: calls.append("disconnect"))
        robot._disconnect_devices()
        assert calls == ["disconnect"]

    def test_a_half_open_bus_is_closed_when_the_driver_would_refuse(self):
        """``is_connected`` false on the robot, true on the bus: close the port."""
        closed: list[bool] = []
        robot = _bare_robot()
        robot.robot = SimpleNamespace(
            is_connected=False,
            bus=SimpleNamespace(is_connected=True, disconnect=lambda disable_torque: closed.append(disable_torque)),
            cameras={},
        )
        robot._disconnect_devices()
        assert closed == [False]

    def test_open_cameras_are_closed(self):
        """A partly-open camera set is walked, and only open cameras closed."""
        closed: list[str] = []
        robot = _bare_robot()
        robot.robot = SimpleNamespace(
            bus=None,
            cameras={
                "front": SimpleNamespace(is_connected=True, disconnect=lambda: closed.append("front")),
                "wrist": SimpleNamespace(is_connected=False, disconnect=lambda: closed.append("wrist")),
            },
        )
        robot._close_open_devices()
        assert closed == ["front"]


class TestNoTeardownReadOutlivesItsAssignment:
    """Derived: the invariant ``__init__``'s own comment claims.

    ``__init__`` says its mesh attributes are set before the bring-up "so
    ``cleanup()``/``__del__`` never see an ``AttributeError`` if construction
    fails partway through". That is a claim about *every* attribute teardown
    reads, so it is graded here against the attributes teardown actually reads
    rather than against a list -- an attribute assigned after the bring-up and
    read during teardown fails this whether or not anyone remembers to add it.
    """

    def test_every_unprotected_teardown_read_is_assigned_before_the_bring_up(self):
        """Established in the closure, or assigned before the statement that can raise."""
        boundary = _bringup_line()
        assignments = _init_assignments()
        offenders = {
            attr: (method, assignments.get(attr))
            for attr, method in _unprotected_state_reads().items()
            if assignments.get(attr) is None or assignments[attr] >= boundary
        }
        assert offenders == {}, (
            "teardown reads instance state that a failed bring-up may not have "
            f"assigned: {offenders}. Either read it as getattr(self, name, "
            f"default) or assign it before the {_BRINGUP_CALL} call "
            f"(__init__ line {boundary})."
        )

    def test_the_survey_reaches_the_device_readers(self):
        """Non-vacuity: an empty closure would make the rule above pass for free."""
        closure = _teardown_closure()
        assert {"cleanup", "_disconnect_devices", "_close_open_devices"} <= closure

    def test_the_survey_finds_real_state_and_excludes_inherited_methods(self):
        """Non-vacuity: the read set is non-empty and holds no bound method."""
        reads = _unprotected_state_reads()
        assert "_shutdown_event" in reads, f"survey found no known state: {sorted(reads)}"
        assert "stop_teleoperate" not in reads, "an inherited method leaked in as state"

    def test_the_lazily_initialised_teleop_state_is_exempt_because_it_is_assigned(self):
        """The exemption is graded, not free.

        ``TeleopMixin._ensure_teleop_state`` is why the ``_teleop_*`` family is
        not reported: the closure assigns it. Drop that initializer and the
        family stops being exempt, which is the correct outcome rather than a
        silently widening allowance.
        """
        established = _closure_established()
        assert {"_teleops", "_teleop_running"} <= established
        assert "_teleops" not in _unprotected_state_reads()
