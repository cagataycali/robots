"""set_gripper / move_to heal the ``robot_name``-first positional misbind.

Historical signature:
    def set_gripper(robot_name=None, state=None, steps=12)
    def move_to(robot_name=None, position=None, ...)

Sibling primitives put the payload first (send_action, set_joint_positions),
and the module's own docstrings use ``set_gripper("close") -> move_to(...)``,
so the natural positional call misbinds the payload to ``robot_name``. The
canonicalizers in motion_primitives_base.py repair the common slip without
touching the signature.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module", autouse=True)
def _mujoco_gl() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")


def _robot():
    from strands_robots import Robot

    return Robot("so101", mesh=False)


def test_set_gripper_positional_state_open_runs_the_gripper() -> None:
    """``set_gripper("open")`` should drive the gripper, not emit ``got None``."""
    r = _robot()
    out = r.set_gripper("open")
    assert out["status"] == "success", out
    text = out["content"][0]["text"]
    assert "commanded open" in text, text


def test_set_gripper_positional_state_close_runs_the_gripper() -> None:
    """``set_gripper("close")`` — the exact idiom from the mujoco docstring."""
    r = _robot()
    out = r.set_gripper("close")
    assert out["status"] == "success", out
    text = out["content"][0]["text"]
    assert "commanded close" in text, text


def test_set_gripper_keyword_form_still_works() -> None:
    """The documented ``set_gripper(state="open")`` form is unchanged."""
    r = _robot()
    out = r.set_gripper(state="open")
    assert out["status"] == "success", out


def test_set_gripper_non_state_positional_still_refuses() -> None:
    """A positional that is not ``open``/``close`` is NOT auto-swapped.

    The canonicalizer only heals the unambiguous misbind. ``set_gripper(0.5)``
    still hits the shared validator with ``state=None`` and the normal
    refusal, so no silent-success window opens on garbage input.
    """
    r = _robot()
    out = r.set_gripper(0.5)
    assert out["status"] == "error", out


def test_set_gripper_explicit_robot_name_and_state_is_a_no_op_swap() -> None:
    """Explicit ``robot_name="so101", state="open"`` must be untouched."""
    r = _robot()
    out = r.set_gripper(robot_name="so101", state="open")
    assert out["status"] == "success", out


def test_move_to_positional_three_vector_reaches_target() -> None:
    """``move_to([x, y, z])`` should solve IK, not refuse with 'requires position'."""
    r = _robot()
    out = r.move_to([0.2, 0.0, 0.1])
    assert out["status"] == "success", out
    text = out["content"][0]["text"]
    assert "reached" in text, text


def test_move_to_keyword_form_still_works() -> None:
    r = _robot()
    out = r.move_to(position=[0.2, 0.0, 0.1])
    assert out["status"] == "success", out


def test_move_to_positional_wrong_length_still_refuses() -> None:
    """A 2-vector or 4-vector positional is NOT a 3D target — don't auto-swap."""
    r = _robot()
    out = r.move_to([0.2, 0.0])
    assert out["status"] == "error", out


def test_canonicalize_set_gripper_args_pure() -> None:
    """Direct test of the swap predicate (no world needed)."""
    from strands_robots.simulation.motion_primitives_base import MotionPrimitivesCore

    canon = MotionPrimitivesCore._canonicalize_set_gripper_args
    assert canon("open", None) == (None, "open")
    assert canon("close", None) == (None, "close")
    # state already set -> no swap
    assert canon("open", "close") == ("open", "close")
    # robot_name is a real robot name -> no swap
    assert canon("so101", None) == ("so101", None)
    # explicit kwargs -> no swap
    assert canon(None, "open") == (None, "open")


def test_canonicalize_move_to_args_pure() -> None:
    from strands_robots.simulation.motion_primitives_base import MotionPrimitivesCore

    canon = MotionPrimitivesCore._canonicalize_move_to_args
    assert canon([0.1, 0.2, 0.3], None) == (None, [0.1, 0.2, 0.3])
    assert canon((0.1, 0.2, 0.3), None) == (None, [0.1, 0.2, 0.3])
    # position already set -> no swap
    assert canon([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]) == ([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    # not length-3 -> no swap
    assert canon([0.1, 0.2], None) == ([0.1, 0.2], None)
    # robot_name string -> no swap
    assert canon("so101", None) == ("so101", None)
    # bool is not a real number for our purposes
    assert canon([True, False, True], None) == ([True, False, True], None)
