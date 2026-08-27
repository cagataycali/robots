"""A whole-arm reading answers for every configured joint, or names the gap.

``MotorController.read_all_positions`` skips a motor whose reply did not
verify, so it returns a subset of ``motor_configs`` carrying no record of what
fell out. Both tool actions that consume it guarded only the all-empty case, so
a bus with one dead servo reported a five-of-six reading as the arm's positions
and persisted it as a named pose.

The two extremes were already pinned - ``test_pose_tool_read_all_formats_every_motor``
drives a bus where every servo answers and
``test_pose_tool_read_all_reports_failure_without_responses`` one where none
does - and those are exactly the two buses on which a partial-drop guard and
its absence are indistinguishable. The mixed bus below is the discriminating
one.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest
import serial

import strands_robots.tools.pose_tool as pose_tool_module
from strands_robots.tools.pose_tool import PoseManager, pose_tool
from tests.tools.conftest import ReadingSerial, position_packet

# wrist_flex, id 4 of the six SO-101 joints. Chosen mid-chain rather than last
# so the reading it falls out of is not merely shorter than expected.
SILENT_JOINT = "wrist_flex"
SILENT_ID = 4


class OneSilentMotor(ReadingSerial):
    """A bus where every servo answers except one - a single dead joint.

    ``ReadingSerial`` answers as whichever motor the outgoing packet addressed,
    which is what lets exactly one id go quiet while the rest of the arm stays
    live.
    """

    def read(self, n: int = 1) -> bytes:
        asked = self.writes[-1][2] if self.writes else 0x01
        if asked == SILENT_ID:
            return b""
        return position_packet(motor_id=asked)


@pytest.fixture
def one_silent_motor(monkeypatch):
    """Patch ``serial.Serial`` with a bus whose ``SILENT_ID`` never replies."""
    instances: list[OneSilentMotor] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> OneSilentMotor:
        bus = OneSilentMotor(port, baudrate, timeout)
        instances.append(bus)
        return bus

    monkeypatch.setattr(serial, "Serial", _ctor)
    return instances


# A second bus, because the order of a one-element gap is not observable and so
# would leave the helper's sort ungraded.
TWO_SILENT_JOINTS = ("shoulder_pan", "wrist_flex")
TWO_SILENT_IDS = (1, 4)


class TwoSilentMotors(ReadingSerial):
    """A bus with two dead joints, one early and one mid-chain."""

    def read(self, n: int = 1) -> bytes:
        asked = self.writes[-1][2] if self.writes else 0x01
        if asked in TWO_SILENT_IDS:
            return b""
        return position_packet(motor_id=asked)


@pytest.fixture
def two_silent_motors(monkeypatch):
    """Patch ``serial.Serial`` with a bus whose two ``TWO_SILENT_IDS`` never reply."""

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> TwoSilentMotors:
        return TwoSilentMotors(port, baudrate, timeout)

    monkeypatch.setattr(serial, "Serial", _ctor)


def _text(result: dict) -> str:
    return " ".join(block["text"] for block in result["content"] if "text" in block)


def _json(result: dict) -> dict | None:
    return next((block["json"] for block in result["content"] if "json" in block), None)


class TestThePremisesTheMixedBusRestsOn:
    """What the fixture and the arm must be for the cases below to mean anything."""

    def test_the_arm_configures_more_joints_than_the_silent_one(self, one_silent_motor) -> None:
        """A partial reading is only distinguishable if several joints answer."""
        controller = pose_tool_module.MotorController("/dev/ttyTEST")
        assert SILENT_JOINT in controller.motor_configs
        assert len(controller.motor_configs) > 2

    def test_the_silent_id_is_the_one_the_named_joint_uses(self, one_silent_motor) -> None:
        """The fixture silences the joint the assertions below name."""
        controller = pose_tool_module.MotorController("/dev/ttyTEST")
        assert controller.motor_configs[SILENT_JOINT]["id"] == SILENT_ID

    def test_the_reading_really_loses_exactly_that_joint(self, cwd_tmp, one_silent_motor) -> None:
        """Every other joint still answers, so the gap is one dead servo."""
        controller = pose_tool_module.MotorController("/dev/ttyTEST")
        connected, _ = controller.connect()
        assert connected
        positions = controller.read_all_positions()
        assert set(controller.motor_configs) - set(positions) == {SILENT_JOINT}


class TestAnIncompleteReadingIsReportedAsOne:
    """read_all over a bus with a dead servo does not present a partial arm."""

    def test_read_all_refuses_a_partial_reading(self, cwd_tmp, one_silent_motor) -> None:
        result = pose_tool(action="read_all", robot_id="hw_arm", port="/dev/ttyTEST")
        assert result["status"] == "error"

    def test_read_all_names_the_joint_that_did_not_answer(self, cwd_tmp, one_silent_motor) -> None:
        result = pose_tool(action="read_all", robot_id="hw_arm", port="/dev/ttyTEST")
        assert SILENT_JOINT in _text(result)

    def test_read_all_reports_how_much_of_the_arm_it_covered(self, cwd_tmp, one_silent_motor) -> None:
        result = pose_tool(action="read_all", robot_id="hw_arm", port="/dev/ttyTEST")
        controller = pose_tool_module.MotorController("/dev/ttyTEST")
        assert f"5 of {len(controller.motor_configs)}" in _text(result)

    def test_several_silent_joints_are_named_in_a_stable_order(self, cwd_tmp, two_silent_motors) -> None:
        """Sorted, so the report does not reorder between two runs of one fault."""
        payload = _json(pose_tool(action="read_all", robot_id="hw_arm", port="/dev/ttyTEST"))
        assert payload is not None
        assert payload["unread"] == sorted(TWO_SILENT_JOINTS)

    def test_read_all_still_carries_the_positions_that_did_arrive(self, cwd_tmp, one_silent_motor) -> None:
        """A caller diagnosing a dead servo needs the rest of the arm."""
        payload = _json(pose_tool(action="read_all", robot_id="hw_arm", port="/dev/ttyTEST"))
        assert payload is not None
        assert len(payload["positions"]) == 5
        assert payload["unread"] == [SILENT_JOINT]


class TestAnIncompletePoseIsNotPersisted:
    """store_pose refuses, because a stored pose is a durable named posture."""

    def test_store_pose_refuses_a_partial_arm(self, cwd_tmp, one_silent_motor) -> None:
        result = pose_tool(action="store_pose", robot_id="hw_arm", port="/dev/ttyTEST", pose_name="home")
        assert result["status"] == "error"

    def test_store_pose_names_the_joint_and_the_consequence(self, cwd_tmp, one_silent_motor) -> None:
        result = pose_tool(action="store_pose", robot_id="hw_arm", port="/dev/ttyTEST", pose_name="home")
        text = _text(result)
        assert SILENT_JOINT in text
        assert "promises the whole arm" in text

    def test_several_silent_joints_are_named_in_a_stable_order(self, cwd_tmp, two_silent_motors) -> None:
        """The refusal lists the joints sorted, matching the reading's own report."""
        result = pose_tool(action="store_pose", robot_id="hw_arm", port="/dev/ttyTEST", pose_name="home")
        assert ", ".join(sorted(TWO_SILENT_JOINTS)) in _text(result)

    def test_nothing_reaches_the_pose_store(self, cwd_tmp, one_silent_motor) -> None:
        """The durable artifact is what makes this one a refusal rather than a report."""
        pose_tool(action="store_pose", robot_id="hw_arm", port="/dev/ttyTEST", pose_name="home")
        assert PoseManager("hw_arm").get_pose("home") is None

    def test_a_later_load_cannot_find_a_pose_that_was_never_stored(self, cwd_tmp, one_silent_motor) -> None:
        """Pre-fix this drove five joints and reported the full pose reached."""
        pose_tool(action="store_pose", robot_id="hw_arm", port="/dev/ttyTEST", pose_name="home")
        result = pose_tool(action="load_pose", robot_id="hw_arm", port="/dev/ttyTEST", pose_name="home")
        assert result["status"] == "error"
        assert "Moved to pose" not in _text(result)


class TestAHealthyArmIsUnchanged:
    """Every joint answering must behave exactly as it did before."""

    def test_read_all_still_succeeds_for_the_whole_arm(self, cwd_tmp, reading_serial) -> None:
        result = pose_tool(action="read_all", robot_id="hw_arm", port="/dev/ttyTEST")
        assert result["status"] == "success"
        assert "Current robot positions" in _text(result)

    def test_read_all_carries_no_unread_key_when_nothing_is_unread(self, cwd_tmp, reading_serial) -> None:
        payload = _json(pose_tool(action="read_all", robot_id="hw_arm", port="/dev/ttyTEST"))
        assert payload is not None
        assert "unread" not in payload

    def test_store_pose_still_persists_a_complete_arm(self, cwd_tmp, reading_serial) -> None:
        controller = pose_tool_module.MotorController("/dev/ttyTEST")
        result = pose_tool(action="store_pose", robot_id="hw_arm", port="/dev/ttyTEST", pose_name="home")
        assert result["status"] == "success"
        stored = PoseManager("hw_arm").get_pose("home")
        assert stored is not None
        assert set(stored.positions) == set(controller.motor_configs)


class TestTheAllSilentBusKeepsItsOwnDiagnosis:
    """A bus where nothing answers is a different fault and reads as one."""

    def test_read_all_reports_the_original_total_failure(self, cwd_tmp, fake_serial) -> None:
        result = pose_tool(action="read_all", robot_id="hw_arm", port="/dev/ttyTEST")
        assert result["status"] == "error"
        assert "Failed to read positions" in _text(result)

    def test_store_pose_reports_the_original_total_failure(self, cwd_tmp, fake_serial) -> None:
        result = pose_tool(action="store_pose", robot_id="hw_arm", port="/dev/ttyTEST", pose_name="home")
        assert result["status"] == "error"
        assert "Failed to read current positions" in _text(result)


class TestBothReadersShareOneAccountOfCompleteness:
    """Derived, so read_all and store_pose cannot drift apart on what is complete."""

    def test_the_helper_has_exactly_one_definition(self) -> None:
        source = pathlib.Path(inspect.getfile(pose_tool_module)).read_text()
        defs = [
            node.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "_joints_that_did_not_answer"
        ]
        assert defs == ["_joints_that_did_not_answer"]

    def test_every_whole_arm_reader_consults_it(self) -> None:
        """A reader of read_all_positions that judges completeness must use the helper."""
        source = pathlib.Path(inspect.getfile(pose_tool_module)).read_text()
        tool = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "pose_tool"
        )
        readers = sum(
            1
            for node in ast.walk(tool)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_all_positions"
        )
        consults = sum(
            1
            for node in ast.walk(tool)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_joints_that_did_not_answer"
        )
        assert readers == 2, readers
        assert consults == readers, (consults, readers)
