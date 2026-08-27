"""Regression: ``rt/lowcmd`` gains come from the vendor's per-joint table.

``_build_lowcmd_from_action`` fills ``kp``/``kd`` for any joint the caller did
not give explicit gains for.  Those defaults used to be a single pair applied to
all 29 joints (``kp=25.0``, ``kd=0.5``), which cannot reproduce the gain set the
G1's low-level position mode is tuned against: the vendor's own ``rt/lowcmd``
reference ships 29-entry ``Kp``/``Kd`` lists taking three and two distinct
values respectively, because the joints do not carry comparable loads.  Both
knees are the stiffest entries at ``kp=100, kd=2`` and they are the joints that
hold a standing biped up.

The firmware validates ``crc``, ``mode_machine`` and the Enable byte; it does
not validate gains.  So an under-gained frame is accepted and the publish
reports success, leaving each joint's closed-loop stiffness at whatever was
sent with no status that says otherwise.  That is the class of defect these
cells exist to catch: wrong on the wire, silent at the API.

Layering, and why it is this way:

* The vendor's numbers are stated **locally** below rather than imported from the
  driver, so the value cells are an independent oracle instead of a tautology.
  A test that reads the constant it grades follows any edit to that constant and
  cannot detect a wrong value -- which is exactly how the single-pair default
  reached the wire past a test whose docstring claimed to pin it.
* The contract cells need no ``unitree_sdk2py``, so ``call-test-lint`` grades
  them in CI, where the SDK is not installed.  ``unitree-sdk2`` is not a
  declared dependency of this project, so a contract asserted only behind
  ``skipif(not _HAS_SDK)`` is asserted by nothing in CI.
* The wire cells, which do need the SDK to build a real ``LowCmd_``, confirm the
  contract cells describe what actually lands on the topic.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest

from strands_robots.drivers.g1 import (
    _G1_JOINT_INDEX,
    _G1_NAMED_JOINTS,
    _SDK_KD,
    _SDK_KP,
    _build_lowcmd_from_action,
    _build_zero_torque_lowcmd,
)

# The vendor's gain lists for this robot, transcribed from the module scope of
# ``unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py``.  Stated
# here so these cells grade the driver against the reference rather than against
# themselves.
VENDOR_KP: tuple[float, ...] = (
    60.0,
    60.0,
    60.0,
    100.0,
    40.0,
    40.0,  # left leg: hip p/r/y, knee, ankle p/r
    60.0,
    60.0,
    60.0,
    100.0,
    40.0,
    40.0,  # right leg
    60.0,
    40.0,
    40.0,  # waist: yaw, roll, pitch
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,  # left arm
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,  # right arm
)
VENDOR_KD: tuple[float, ...] = (
    1.0,
    1.0,
    1.0,
    2.0,
    1.0,
    1.0,  # left leg
    1.0,
    1.0,
    1.0,
    2.0,
    1.0,
    1.0,  # right leg
    1.0,
    1.0,
    1.0,  # waist
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,  # left arm
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,  # right arm
)

_KNEE_SLOTS = (3, 9)
_SLOT_NAME = {slot: name for name, slot in _G1_JOINT_INDEX.items()}

try:  # pragma: no cover - environment probe
    import unitree_sdk2py.idl.default as _sdk_default

    _HAS_SDK = _sdk_default is not None
except Exception:  # pragma: no cover - environment probe
    _HAS_SDK = False


def _slot_ids() -> list[str]:
    """Parametrize ids that name the joint, so a failure reads as the joint."""
    return [f"{slot}-{_SLOT_NAME.get(slot, '?')}" for slot in range(_G1_NAMED_JOINTS)]


class TestTheGainTableIsTheVendorReference:
    """Every slot's default gains equal the vendor's value for that slot."""

    def test_the_table_covers_exactly_the_named_joints(self) -> None:
        """One entry per commanded joint, aligned with the Enable-byte bound.

        ``_build_lowcmd_from_action`` indexes the table by the slot a joint
        name resolved to, so a table shorter than the joint map is an
        ``IndexError`` on the last joints, and a longer one carries entries no
        joint can reach.  Tying the length to ``_G1_NAMED_JOINTS`` means a
        joint added to the map later moves the table in the same edit.
        """
        assert len(_SDK_KP) == _G1_NAMED_JOINTS
        assert len(_SDK_KD) == _G1_NAMED_JOINTS
        assert len(VENDOR_KP) == _G1_NAMED_JOINTS  # the local oracle, same width

    @pytest.mark.parametrize("slot", range(_G1_NAMED_JOINTS), ids=_slot_ids())
    def test_each_slot_carries_the_vendor_gains(self, slot: int) -> None:
        """Graded per slot so a wrong entry names the joint it belongs to."""
        assert _SDK_KP[slot] == pytest.approx(VENDOR_KP[slot])
        assert _SDK_KD[slot] == pytest.approx(VENDOR_KD[slot])

    def test_both_knees_are_the_stiffest_entries(self) -> None:
        """The load-bearing joints hold the table maximum.

        Called out on its own because it is the consequence that matters: a
        knee is what a standing G1 holds itself up with, and it is the slot a
        scalar default understiffens most.
        """
        for slot in _KNEE_SLOTS:
            assert _SLOT_NAME[slot].endswith("knee")
            # Strictly greater than the table minimum, not merely equal to the
            # maximum: on a flattened table every entry ties the maximum, so an
            # equality-only assertion would pass on exactly the shape this cell
            # exists to reject.
            assert _SDK_KP[slot] == max(_SDK_KP) > min(_SDK_KP)
            assert _SDK_KD[slot] == max(_SDK_KD) > min(_SDK_KD)


class TestTheGainsAreGenuinelyPerJoint:
    """The table cannot be replaced by any single pair of numbers.

    This is the cell that fails if the per-slot table is ever collapsed back
    into a scalar -- including a scalar chosen from inside the table's own
    range, which no equality-against-a-constant test would notice.
    """

    def test_the_table_holds_more_than_one_value(self) -> None:
        assert len(set(_SDK_KP)) > 1, "a single kp cannot be the vendor's table"
        assert len(set(_SDK_KD)) > 1, "a single kd cannot be the vendor's table"

    def test_the_distinct_values_are_the_vendor_s(self) -> None:
        """Three stiffnesses and two dampings, matching the reference."""
        assert sorted(set(_SDK_KP)) == sorted(set(VENDOR_KP))
        assert sorted(set(_SDK_KD)) == sorted(set(VENDOR_KD))
        assert len(set(_SDK_KP)) == 3
        assert len(set(_SDK_KD)) == 2

    def test_a_knee_is_stiffer_than_an_arm_by_the_reference_ratio(self) -> None:
        """The spread a scalar erases, stated as the ratio it would erase.

        ``right_elbow`` is an ordinary arm joint; the knee is the extreme. Any
        scalar makes this ratio 1.0.
        """
        knee, arm = _G1_JOINT_INDEX["left_knee"], _G1_JOINT_INDEX["right_elbow"]
        assert _SDK_KP[knee] / _SDK_KP[arm] == pytest.approx(2.5)
        assert _SDK_KD[knee] / _SDK_KD[arm] == pytest.approx(2.0)


class TestTheBuilderIndexesTheTablePerSlot:
    """The builder reads the table by slot, not by taking one entry for all.

    A structural cell, because it is reachable without the SDK and it pins the
    mechanism rather than one sampled value: subscripting the table with the
    resolved slot is what makes the per-joint values reach the wire at all.
    """

    def test_both_gain_tables_are_subscripted_in_the_builder(self) -> None:
        tree = ast.parse(inspect.getsource(_build_lowcmd_from_action))
        subscripted = {
            node.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
        }
        assert "_SDK_KP" in subscripted, "kp default is not indexed by slot"
        assert "_SDK_KD" in subscripted, "kd default is not indexed by slot"


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
class TestTheWireFrameCarriesTheSlotsOwnGains:
    """The frame that reaches ``rt/lowcmd`` carries each slot's own gains."""

    def test_every_slot_lands_its_vendor_gains(self) -> None:
        """A scalar target for all 29 joints, checked slot by slot."""
        action: dict[str, Any] = {name: 0.0 for name in _G1_JOINT_INDEX}
        cmd, err = _build_lowcmd_from_action(action, mode_machine=9)
        assert err is None
        assert cmd is not None
        for slot in range(_G1_NAMED_JOINTS):
            motor = cmd.motor_cmd[slot]
            assert motor.kp == pytest.approx(VENDOR_KP[slot]), _SLOT_NAME[slot]
            assert motor.kd == pytest.approx(VENDOR_KD[slot]), _SLOT_NAME[slot]

    def test_a_knee_and_an_arm_in_one_frame_differ(self) -> None:
        """Two joints, one frame, two gain pairs - the scalar's failure mode."""
        cmd, err = _build_lowcmd_from_action({"left_knee": 0.2, "right_elbow": -0.2}, mode_machine=9)
        assert err is None
        assert cmd is not None
        knee = cmd.motor_cmd[_G1_JOINT_INDEX["left_knee"]]
        arm = cmd.motor_cmd[_G1_JOINT_INDEX["right_elbow"]]
        assert knee.kp == pytest.approx(100.0)
        assert arm.kp == pytest.approx(40.0)
        assert knee.kp != arm.kp

    def test_a_partially_supplied_gain_falls_back_to_that_slot(self) -> None:
        """``kp`` supplied, ``kd`` omitted - the omitted one is the slot's own.

        The fallback has to stay per-slot in the partial case too, or a caller
        tuning one term silently flattens the other.  Graded here rather than
        among the over-reach guards because it fails on the flat-default shape,
        so it grades the fix rather than protecting what the fix must not move.
        """
        cmd, err = _build_lowcmd_from_action({"left_knee": {"q": 0.1, "kp": 7.0}}, mode_machine=9)
        assert err is None
        assert cmd is not None
        motor = cmd.motor_cmd[_G1_JOINT_INDEX["left_knee"]]
        assert motor.kp == pytest.approx(7.0)
        assert motor.kd == pytest.approx(VENDOR_KD[_G1_JOINT_INDEX["left_knee"]])


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
class TestWhatTheTableDoesNotChange:
    """Over-reach guards: the table is a default, and only a default.

    Both cells hold on the flat-default shape as well as on the table, which is
    what makes them guards rather than regression cells - they fail only if the
    change reached somewhere it should not have.
    """

    def test_a_supplied_gain_still_wins(self) -> None:
        """An explicit ``kp``/``kd`` overrides the table, as before."""
        cmd, err = _build_lowcmd_from_action({"left_knee": {"q": 0.1, "kp": 7.0, "kd": 0.25}}, mode_machine=9)
        assert err is None
        assert cmd is not None
        motor = cmd.motor_cmd[_G1_JOINT_INDEX["left_knee"]]
        assert motor.kp == pytest.approx(7.0)
        assert motor.kd == pytest.approx(0.25)

    def test_the_zero_torque_frame_stays_soft(self) -> None:
        """A stop frame zeroes gains; the table must not leak into it.

        ``_build_zero_torque_lowcmd`` is what "soft" looks like on the wire.
        Its gains are zero by intent, so a table applied there would make a
        stop stiff.
        """
        cmd, err = _build_zero_torque_lowcmd(mode_machine=9)
        assert err is None
        assert cmd is not None
        assert all(motor.kp == 0.0 for motor in cmd.motor_cmd)
        assert all(motor.kd == 0.0 for motor in cmd.motor_cmd)
