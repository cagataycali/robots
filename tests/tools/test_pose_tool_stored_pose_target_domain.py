# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the degree-valued targets ``load_pose`` drives from disk.

A stored pose reaches a servo through the same
``MotorController.degrees_to_position`` as an argument target: the value is
clamped into the joint's configured ``range`` and scaled onto the 12-bit
``Goal_Position`` register. That clamp is why an off-domain stored target is
dangerous rather than merely wrong, and it is what this module measures the
refusals against:

* **A stored target outside the travel shared an encoding with an end stop.**
  ``TestWhyAStoredTargetIsRefused`` shows ``999`` and ``nan`` both converting to
  ``Goal_Position`` 4095 - a full-travel command to the mechanical limit.

* **And the caller was told the arm went where the file said.** ``load_pose``
  reports ``"Moved to pose '<name>'"`` and echoes ``target_positions`` straight
  from the artifact, so the pre-guard envelope quoted ``999`` for a move to
  +180. ``TestTheBusIsNotTouched`` pins the replacement: a refused pose opens no
  port and writes nothing.

The bound is not restated here. Every stored position is delegated to
:func:`~strands_robots.tools.pose_tool._joint_target_error`, the same authority
the ``position`` / ``positions`` / ``delta`` arguments are held to, so a stored
target and an argument target cannot drift apart. That split is asserted in
``TestTheBoundsHaveOneAuthority`` rather than left to convention.

``PoseManager.validate_pose`` is not this check, and
``TestTheDeferralItReplacesCannotFire`` measures why: it consults the pose's own
optional ``safety_bounds`` and answers "No safety bounds defined" when the field
is absent, which it is for every pose this tool writes.

Every test that reaches the motor path takes a serial stand-in and passes an
explicit fake ``port``: ``pose_tool``'s ``port`` defaults to ``/dev/ttyACM0``,
so a test that omits it drives whatever arm is plugged into the machine running
the suite.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.pose_tool as pose_tool_module
from strands_robots.tools.pose_tool import (
    MotorController,
    PoseManager,
    RobotPose,
    _joint_target_error,
    _stored_pose_target_error,
    pose_tool,
)

_PORT = "/dev/fake-stored-pose"
_ROBOT = "stored_pose_arm"
_POSE = "bench"

# A joint with a symmetric configured travel, so a refusal names a bound that is
# obviously not the whole float line.
_JOINT = "shoulder_pan"
_JOINT_RANGE = (-180, 180)

# A pose every joint of which is inside its travel: the reading the tool itself
# would have persisted, and the control this change must not refuse.
_USABLE_POSITIONS: dict[str, float] = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 5.0,
    "elbow_flex": -20.0,
    "wrist_flex": 0.0,
    "wrist_roll": 45.0,
    "gripper": 50.0,
}

# Stored targets no joint can be driven to, each for one of the two reasons the
# shared domain gives: not a finite number, or outside the configured travel.
_UNUSABLE_TARGETS: tuple[Any, ...] = (
    math.nan,
    math.inf,
    -math.inf,
    999.0,
    -999.0,
    180.5,
    5000,
    True,
    "90",
    [90],
    10**400,
)


def _write_pose_file(root: Path, positions: dict[str, Any], name: str = _POSE) -> Path:
    """Persist a pose the way :meth:`PoseManager._save_poses` does.

    The pose file is a plain JSON artifact under the storage directory and
    ``_load_poses`` reads it back through ``RobotPose.from_dict`` without
    checking anything, so writing it directly is the route a retuned or
    hand-edited pose takes.

    Args:
        root: The directory ``PoseManager`` resolves its storage under (cwd).
        positions: The stored motor targets.
        name: The pose name.

    Returns:
        The path written.
    """
    pose_file = root / ".strands_robots" / "poses" / f"{_ROBOT}_poses.json"
    pose_file.parent.mkdir(parents=True, exist_ok=True)
    document = {
        name: {
            "name": name,
            "positions": positions,
            "timestamp": 0.0,
            "description": None,
            "safety_bounds": None,
        }
    }
    pose_file.write_text(json.dumps(document))
    return pose_file


def _load(**kwargs: Any) -> dict[str, Any]:
    """Invoke ``load_pose`` through one funnel with an explicit fake port."""
    return pose_tool(action="load_pose", robot_id=_ROBOT, port=_PORT, pose_name=_POSE, **kwargs)


def _texts(result: dict[str, Any]) -> str:
    """Concatenate every ``text`` field of a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _goal_positions(instances: list[Any]) -> list[int]:
    """Every ``Goal_Position`` that reached the bus, decoded from the packets.

    A goal write is ``INST_WRITE`` (``0x03``) whose first parameter is the
    ``Goal_Position`` address (``0x2A``), value little-endian after it. Reads
    share the bus, so the payload is what distinguishes a command from a query.

    Args:
        instances: The recording serial stand-ins the fixture handed out.

    Returns:
        The commanded goal positions, in write order.
    """
    goals: list[int] = []
    for fake in instances:
        for packet in fake.writes:
            if len(packet) >= 9 and packet[4] == 0x03 and packet[5] == 0x2A:
                goals.append(packet[6] | (packet[7] << 8))
    return goals


class TestWhyAStoredTargetIsRefused:
    """The domain is justified by what the conversion does with the value."""

    @pytest.mark.parametrize("stored", [999.0, 5000, math.inf, math.nan])
    def test_a_stored_target_over_the_travel_encodes_as_the_end_stop(self, stored: Any) -> None:
        """Each shares one encoding with a full-travel command to the limit."""
        assert MotorController(_PORT).degrees_to_position(_JOINT, stored) == 4095

    def test_nan_reaches_the_limit_through_the_clamp_itself(self) -> None:
        """``min(max_deg, nan)`` returns ``max_deg``, so the guard fabricates it."""
        assert min(_JOINT_RANGE[1], math.nan) == _JOINT_RANGE[1]

    def test_the_end_stop_encoding_is_indistinguishable_from_a_deliberate_limit(self) -> None:
        controller = MotorController(_PORT)
        assert controller.degrees_to_position(_JOINT, 999.0) == controller.degrees_to_position(_JOINT, _JOINT_RANGE[1])


class TestAStoredTargetOutsideTheTravelIsRefused:
    """The regression: the stored value is held to the joint's travel."""

    @pytest.mark.parametrize("stored", _UNUSABLE_TARGETS)
    def test_every_unusable_stored_target_is_refused(self, stored: Any, reading_serial, cwd_tmp) -> None:
        _write_pose_file(cwd_tmp, {**_USABLE_POSITIONS, _JOINT: stored})
        result = _load(smooth=False)
        assert result["status"] == "error", _texts(result)

    def test_the_refusal_names_the_pose_the_joint_and_the_bound(self, reading_serial, cwd_tmp) -> None:
        """A stored value is corrected in a file, so the file must be named."""
        _write_pose_file(cwd_tmp, {**_USABLE_POSITIONS, _JOINT: 999.0})
        text = _texts(_load(smooth=False))
        assert "load_pose" in text
        assert repr(_POSE) in text
        assert repr(_JOINT) in text
        assert f"[{_JOINT_RANGE[0]}, {_JOINT_RANGE[1]}]" in text
        assert "999" in text

    def test_the_label_itself_names_both_the_pose_and_the_joint(self, reading_serial, cwd_tmp) -> None:
        """The shared message names the joint too, so grade the label alone.

        Everything before ``must be`` is what this module contributes; the rest
        is the shared domain's wording. Reading only the whole message would let
        the label stop naming the joint without a test noticing, because the
        travel clause names it again.
        """
        _write_pose_file(cwd_tmp, {**_USABLE_POSITIONS, _JOINT: 999.0})
        label = _texts(_load(smooth=False)).split(" must be")[0]
        assert repr(_POSE) in label
        assert repr(_JOINT) in label

    def test_the_interpolating_path_refuses_the_same_stored_target(self, reading_serial, cwd_tmp) -> None:
        """``smooth`` chooses how the arm gets there, not whether it may."""
        _write_pose_file(cwd_tmp, {**_USABLE_POSITIONS, _JOINT: 999.0})
        assert _load(smooth=True, steps=3, step_delay=0.0)["status"] == "error"


class TestTheBusIsNotTouched:
    """A refused pose reaches no servo and opens no port."""

    def test_no_goal_position_is_written(self, reading_serial, cwd_tmp) -> None:
        _write_pose_file(cwd_tmp, {**_USABLE_POSITIONS, _JOINT: 999.0})
        _load(smooth=False)
        assert _goal_positions(reading_serial) == []

    def test_the_port_is_never_opened(self, reading_serial, cwd_tmp) -> None:
        """Refused before ``connect``, so a bad pose cannot hold the bus."""
        _write_pose_file(cwd_tmp, {**_USABLE_POSITIONS, _JOINT: 999.0})
        _load(smooth=False)
        assert reading_serial == []

    def test_the_envelope_no_longer_echoes_a_target_that_was_not_commanded(self, reading_serial, cwd_tmp) -> None:
        _write_pose_file(cwd_tmp, {**_USABLE_POSITIONS, _JOINT: 999.0})
        result = _load(smooth=False)
        assert all("json" not in item for item in result["content"])


class TestTheBoundsHaveOneAuthority:
    """The stored-pose check decides no bound of its own."""

    def test_it_delegates_to_the_shared_joint_domain(self) -> None:
        source = inspect.getsource(_stored_pose_target_error)
        assert "_joint_target_error(" in source

    def test_it_restates_no_numeric_bound(self) -> None:
        """A second copy of the travel could disagree with the driven one."""
        function = ast.parse(inspect.getsource(_stored_pose_target_error).lstrip()).body[0]
        assert isinstance(function, ast.FunctionDef)
        body = function.body[1:] if ast.get_docstring(function) else function.body
        numbers = [
            node.value
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Constant) and isinstance(node.value, int | float)
        ]
        assert numbers == []

    def test_the_two_surfaces_agree_on_the_same_value(self) -> None:
        """An argument target and a stored target are one domain."""
        stored = _stored_pose_target_error(RobotPose(name=_POSE, positions={_JOINT: 999.0}, timestamp=0.0))
        argument = _joint_target_error("load_pose", "position", _JOINT, 999.0)
        assert (stored is None) == (argument is None) is False


class TestTheDeferralItReplacesCannotFire:
    """Why ``validate_pose`` was not already covering this."""

    def test_the_tool_never_stores_safety_bounds(self) -> None:
        """Every ``store_pose`` call site omits the field it would need."""
        source = inspect.getsource(pose_tool_module)
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "store_pose"
        ]
        assert calls, "no store_pose call site found to grade"
        assert all(keyword.arg != "safety_bounds" for call in calls for keyword in call.keywords)
        assert all(len(call.args) <= 3 for call in calls)

    def test_validate_pose_passes_a_pose_stored_without_bounds(self, cwd_tmp) -> None:
        pose = RobotPose(name=_POSE, positions={_JOINT: 999.0}, timestamp=0.0)
        assert PoseManager(_ROBOT).validate_pose(pose) == (True, "No safety bounds defined")


class TestUsableStoredPosesStillLoad:
    """The boundary: nothing inside the travel is refused."""

    def test_an_in_range_pose_still_drives_every_joint(self, reading_serial, cwd_tmp) -> None:
        _write_pose_file(cwd_tmp, dict(_USABLE_POSITIONS))
        result = _load(smooth=False)
        assert result["status"] == "success", _texts(result)
        assert len(_goal_positions(reading_serial)) == len(_USABLE_POSITIONS)

    def test_an_in_range_pose_still_reports_its_targets(self, reading_serial, cwd_tmp) -> None:
        _write_pose_file(cwd_tmp, dict(_USABLE_POSITIONS))
        result = _load(smooth=False)
        payload = next(item["json"] for item in result["content"] if "json" in item)
        assert payload["target_positions"] == _USABLE_POSITIONS

    @pytest.mark.parametrize("edge", [_JOINT_RANGE[0], _JOINT_RANGE[1]])
    def test_a_stored_target_at_the_travel_endpoint_still_loads(self, edge: float, reading_serial, cwd_tmp) -> None:
        """The endpoints are reachable positions, not off-domain ones."""
        _write_pose_file(cwd_tmp, {**_USABLE_POSITIONS, _JOINT: float(edge)})
        assert _load(smooth=False)["status"] == "success"

    def test_a_stored_motor_with_no_configured_travel_has_no_bound_to_check(self) -> None:
        """No configured range means no travel to hold a target against."""
        pose = RobotPose(name=_POSE, positions={"no_such_joint": 5000.0}, timestamp=0.0)
        assert _stored_pose_target_error(pose) is None

    def test_an_unknown_motor_is_left_to_the_mover_that_cannot_address_it(self, reading_serial, cwd_tmp) -> None:
        """Still refused, but for having no motor id rather than for its value."""
        _write_pose_file(cwd_tmp, {**_USABLE_POSITIONS, "no_such_joint": 5000.0})
        result = _load(smooth=False)
        assert result["status"] == "error"
        assert "configured travel" not in _texts(result)

    def test_a_stored_motor_with_no_travel_is_still_held_to_finiteness(self, reading_serial, cwd_tmp) -> None:
        """Finiteness needs no range, so the shared domain still applies."""
        _write_pose_file(cwd_tmp, {**_USABLE_POSITIONS, "no_such_joint": math.nan})
        assert _load(smooth=False)["status"] == "error"


class TestReachingHomeStaysOutOfScope:
    """``reset_to_home`` supplies its own literals, and they are in range."""

    def test_every_home_literal_is_inside_its_configured_travel(self) -> None:
        """Read off the literals themselves, so retuning one out of range fires."""
        module = ast.parse(Path(pose_tool_module.__file__).read_text())
        assignments = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "home_positions" for target in node.targets)
        ]
        assert len(assignments) == 1, "reset_to_home no longer supplies one literal pose"
        literals = ast.literal_eval(assignments[0].value)
        assert literals, "the home pose is empty"
        for name, value in literals.items():
            low, high = pose_tool_module._DEFAULT_MOTOR_CONFIGS[name]["range"]
            assert low <= value <= high, (name, value)
