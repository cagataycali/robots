# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests: a joint's configured ``range`` is the scale it is driven on.

``MotorController.motor_configs`` is a per-instance copy of
``_DEFAULT_MOTOR_CONFIGS`` -- built by copying each entry so that retuning one
controller's travel cannot leak into another's, which
``test_pose_tool_target_domain`` pins directly. Retuning is therefore a
supported operation, and the module comment above that table states what has to
hold when someone does it: ``range`` is "consulted twice: by
``degrees_to_position``, which converts a target into a ``Goal_Position``, and
by ``_joint_target_error``, which refuses a target that conversion could not
represent", because "a second copy of these bounds could disagree with the one
the servo is actually driven from".

The conversion selected the gripper's mapping by *name* and hard-coded 0-100 in
place of that joint's bounds, so it read no ``range`` at all for one of the six
joints. On the shipped table the two arithmetics coincide -- the gripper's
default range *is* ``(0, 100)`` -- so the special case changed nothing until a
caller retuned it, and then the servo was driven on a scale the guard was not
checking against. With ``range`` set to ``(0, 50)``, a fully-open command wrote
2047 counts, half of the encoder, and ``position_to_degrees`` read that same
count back as ``50.0``: the write and the read shared the wrong scale, so the
arm reported the value it had been asked for while sitting at half travel.

``TestTheShippedMappingIsUnchanged`` is the other half of the pin. The fix is a
deletion, so the cells that matter most are the ones proving the default table
still maps exactly as it always did -- they pass before and after on purpose.
"""

from __future__ import annotations

import pytest

from strands_robots.tools.pose_tool import _DEFAULT_MOTOR_CONFIGS, MotorController

_PORT = "/dev/fake-arm"

#: ``Goal_Position`` on the Feetech STS/SMS control table, as ``move_motor`` writes it.
_GOAL_POSITION_ADDRESS = 0x2A

#: A retuned travel per joint, narrower or shifted from the shipped default. The
#: gripper's is the one the old name branch ignored; the other five are controls
#: that already honored their bounds and must keep honoring them.
_RETUNED_TRAVEL: dict[str, tuple[int, int]] = {
    "shoulder_pan": (-90, 90),
    "shoulder_lift": (-45, 45),
    "elbow_flex": (-100, 100),
    "wrist_flex": (-60, 60),
    "wrist_roll": (-90, 90),
    "gripper": (0, 50),
}


def _connected(port: str = _PORT) -> MotorController:
    """A controller whose ``serial_conn`` is the patched fake."""
    controller = MotorController(port=port)
    connected, error = controller.connect()
    assert connected, error
    return controller


def _goal_position_written(writes: list[bytes]) -> int:
    """The count in the last ``Goal_Position`` write, decoded off the wire.

    ``FF FF ID LEN INST ADDR LO HI CHK``, low byte first - the STS/SMS order the
    protocol codec emits, which ``test_feetech_status_packet_framing`` grades.
    """
    packet = writes[-1]
    assert packet[5] == _GOAL_POSITION_ADDRESS, f"not a Goal_Position write: {packet.hex(' ')}"
    return packet[6] | (packet[7] << 8)


class TestTheConfiguredTravelIsTheScale:
    """Every joint's own bounds map onto the ends of its encoder."""

    @pytest.mark.parametrize("joint", sorted(_DEFAULT_MOTOR_CONFIGS))
    def test_the_shipped_bounds_of_every_joint_span_the_encoder(self, joint: str) -> None:
        controller = MotorController(port=_PORT)
        low, high = controller.motor_configs[joint]["range"]
        resolution = controller.motor_configs[joint]["resolution"]

        assert controller.degrees_to_position(joint, low) == 0
        assert controller.degrees_to_position(joint, high) == resolution

    @pytest.mark.parametrize("joint", sorted(_RETUNED_TRAVEL))
    def test_a_retuned_travel_is_the_scale_that_reaches_the_wire(self, joint: str, fake_serial) -> None:
        controller = _connected()
        low, high = _RETUNED_TRAVEL[joint]
        controller.motor_configs[joint]["range"] = (low, high)
        resolution = controller.motor_configs[joint]["resolution"]
        writes = fake_serial[0].writes

        assert controller.move_motor(joint, float(low)) is True
        assert _goal_position_written(writes) == 0, f"{joint} at its configured low end is not encoder zero"
        assert controller.move_motor(joint, float(high)) is True
        assert _goal_position_written(writes) == resolution, (
            f"{joint} at its configured high end is not encoder full scale"
        )

    @pytest.mark.parametrize("joint", sorted(_RETUNED_TRAVEL))
    def test_a_reading_is_quoted_on_the_scale_the_joint_is_driven_on(self, joint: str) -> None:
        """The count is derived from the bounds here, not from the writer.

        Asking :meth:`degrees_to_position` for the count and handing it straight
        back to :meth:`position_to_degrees` cannot see this class of defect: both
        directions read the same scale, so a wrong one round-trips perfectly --
        which is why the gripper reported the value it had been asked for while
        sitting at half travel. The count therefore comes from the configured
        bounds independently, so the reader is graded on its own.
        """
        controller = MotorController(port=_PORT)
        low, high = _RETUNED_TRAVEL[joint]
        controller.motor_configs[joint]["range"] = (low, high)
        resolution = controller.motor_configs[joint]["resolution"]

        for fraction, expected in ((0.0, float(low)), (1.0, float(high)), (0.5, (low + high) / 2)):
            counts = round(fraction * resolution)
            assert controller.position_to_degrees(joint, counts) == pytest.approx(expected, abs=0.2)


class TestTheShippedMappingIsUnchanged:
    """The default table maps exactly as it did before the name branch went away."""

    @pytest.mark.parametrize(
        ("joint", "target", "counts"),
        [
            ("gripper", 0.0, 0),
            ("gripper", 25.0, 1023),
            ("gripper", 50.0, 2047),
            ("gripper", 100.0, 4095),
            ("shoulder_pan", -180.0, 0),
            ("shoulder_pan", 0.0, 2047),
            ("shoulder_pan", 180.0, 4095),
            ("wrist_flex", -90.0, 0),
            ("wrist_flex", 0.0, 2047),
            ("wrist_flex", 90.0, 4095),
        ],
    )
    def test_a_target_on_the_default_table_encodes_as_it_always_has(
        self, joint: str, target: float, counts: int
    ) -> None:
        assert MotorController(port=_PORT).degrees_to_position(joint, target) == counts

    def test_the_gripper_is_still_quoted_percent_open_by_default(self) -> None:
        controller = MotorController(port=_PORT)

        assert controller.motor_configs["gripper"]["range"] == (0, 100)
        assert controller.position_to_degrees("gripper", 4095) == pytest.approx(100.0)
        assert controller.position_to_degrees("gripper", 0) == pytest.approx(0.0)
