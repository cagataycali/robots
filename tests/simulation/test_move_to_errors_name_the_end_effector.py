"""Every move_to refusal names the end-effector frame it tried to place.

On a g1 humanoid ``move_to([0.3, 0, 0.5])`` refuses with "unreachable for 'g1'
... move_to drives the arm's position servos" - but the frame it drove was
``g1/left_wrist_roll_link``, the chain-leaf rung of ``discover_ee_frame`` (g1's
only four sites are two IMUs and two feet, so no site rung matches), and the 29
joints it commanded are both legs, the waist and both arms. The success text
already names the frame; the three refusals did not, so a caller who reached
with the wrong hand had nothing to correct from. The frame was in the JSON
payload (``frame``/``frame_type``) all along - this puts it in the text the
model actually reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.simulation.motion_primitives_base import MotionPrimitivesCore


class _Sim(MotionPrimitivesCore):
    pass


TARGET = np.array([0.3, 0.0, 0.5])


def _text(result: dict) -> str:
    return result["content"][0]["text"]


@pytest.mark.parametrize("orientation_tol", [None, 0.1])
def test_unreachable_error_names_the_frame(orientation_tol: float | None) -> None:
    res = _Sim._move_to_unreachable_error(
        "g1",
        TARGET,
        0.015,
        ik_residual=0.1688,
        frame_name="g1/left_wrist_roll_link",
        frame_type="body",
        orientation_tol=orientation_tol,
        ik_orientation_residual=0.5 if orientation_tol else None,
        position_only_residual=None,
        unrestricted_residual=None,
        uncommanded_joints=[],
    )
    assert res["status"] == "error"
    assert "body 'g1/left_wrist_roll_link'" in _text(res)


def test_borrowed_dof_remedy_names_the_frame_not_the_arm() -> None:
    res = _Sim._move_to_unreachable_error(
        "g1",
        TARGET,
        0.015,
        ik_residual=0.1688,
        frame_name="g1/left_wrist_roll_link",
        frame_type="body",
        orientation_tol=None,
        ik_orientation_residual=None,
        position_only_residual=None,
        unrestricted_residual=0.0,
        uncommanded_joints=["g1/floating_base_joint"],
    )
    text = _text(res)
    assert "drives the body 'g1/left_wrist_roll_link'" in text
    assert "the arm's" not in text


def test_not_reached_error_names_the_frame() -> None:
    res = _Sim._move_to_result(
        "so100",
        TARGET,
        0.015,
        200,
        reached=False,
        steps_used=200,
        position_error=0.02,
        ik_residual=0.0,
        ee_pos=[0.3, 0.0, 0.48],
        ee_quat=[1.0, 0.0, 0.0, 0.0],
        frame_name="so100/Wrist_Pitch_Roll",
        frame_type="body",
        orientation_error=None,
        orientation_tol=None,
        ik_orientation_residual=None,
    )
    assert res["status"] == "error"
    assert "EE (body 'so100/Wrist_Pitch_Roll') did not reach" in _text(res)
