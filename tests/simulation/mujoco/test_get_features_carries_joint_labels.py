"""``get_features`` carries the registry's ``joint_labels`` into its sidecar.

harness#520 taught ``get_robot_state`` to render ``'1 (shoulder_pan): …'`` and
ship a sidecar ``joint_labels`` for arms whose upstream MJCF names the joints
by servo id (so101). harness#758 taught ``_world_readiness_sentence`` the same
annotation on the tool_spec hot path. ``get_features`` is the next hop a reader
takes when it asks "what does this robot have?" --- the method whose docstring
on ``base.py`` and whose MuJoCo override docstring sentence 1 both read
"Describe the simulation's joints / actuators / cameras / robots" --- and it
dropped the label map, so an agent reading the json sidecar learned the arm
had six joints named ``1``..``6`` and no semantic hint.

This is NOT the same closed class as:

* harness#712 (``get_observation`` numeric for so101) --- schema contract on
  base.py: observation keys are recorder columns, aliasing them would double
  every dataset column. ``get_features``'s per-robot sidecar is pure discovery
  metadata (``data_config``, ``source``, counts, joint names) --- never read by
  any recorder.
* harness#768 (``robot_action_keys`` returns actuator names on lekiwi) ---
  same column-stability reasoning (``Policy.set_robot_state_keys`` and every
  backend's ``recording.py`` read this list verbatim).

The fix carries the registry ``joint_labels`` through
``_robot_joint_labels(robot)`` into each ``robots.{name}`` entry, mirrors the
annotation ``_world_readiness_sentence`` adds on the text block, and leaves
robots without a registry label entry (``panda``, ``g1``, ``go2``, ...) with an
empty dict so their legacy shape is byte-identical.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots import Robot


_SO101_LABELS = {
    "1": "shoulder_pan",
    "2": "shoulder_lift",
    "3": "elbow_flex",
    "4": "wrist_flex",
    "5": "wrist_roll",
    "6": "gripper",
}


def _sidecar(sim, robot_name: str) -> dict:
    result = sim.get_features(robot_name=robot_name)
    for item in result["content"]:
        if "json" in item:
            return item["json"]["features"]["robots"][robot_name]
    pytest.fail("get_features emitted no JSON sidecar")


def _text(sim, robot_name: str) -> str:
    result = sim.get_features(robot_name=robot_name)
    return next(item["text"] for item in result["content"] if "text" in item)


# --------- so101: labels are carried through the fix --------------------------


def test_so101_sidecar_carries_registry_joint_labels():
    """The registry labels so101 uses on the write side flow into the sidecar.

    harness#520's ``get_robot_state`` sidecar key is reused here, so a reader
    moving between the two methods sees the same label vocabulary. The sidecar
    on ``base.py`` for ``get_robot_state`` is pinned by
    ``test_joint_labels_address_so_arm_joints.py``.
    """
    sim = Robot("so101")

    side = _sidecar(sim, "so101")

    assert side["joint_labels"] == _SO101_LABELS
    # The existing shape is preserved beside the new key.
    assert side["joint_names"] == list(_SO101_LABELS.keys())


def test_so101_text_block_annotates_joints_like_world_readiness_sentence():
    """The text block adds ``joint_labels: 1 (shoulder_pan), ...``.

    Mirrors the shape ``_world_readiness_sentence`` adopted in harness#758 so
    the two agent-first text surfaces read the same way about the same arm.
    """
    sim = Robot("so101")

    text = _text(sim, "so101")

    # One annotated line per robot, keyed off the registry labels.
    assert (
        "joint_labels: 1 (shoulder_pan), 2 (shoulder_lift), 3 (elbow_flex),"
        " 4 (wrist_flex), 5 (wrist_roll), 6 (gripper)" in text
    )


# --------- panda: no labels, legacy shape unchanged ---------------------------


def test_panda_sidecar_carries_empty_label_dict_not_missing_key():
    """A robot without registry labels gets ``{}`` --- the schema is stable.

    An introspecting reader that iterates ``.joint_labels.items()`` on every
    robot in the ``robots`` map does not need to branch on whether the key
    exists.
    """
    sim = Robot("panda")

    side = _sidecar(sim, "panda")

    assert side["joint_labels"] == {}
    # Legacy shape is untouched beside the new key.
    assert side["joint_names"]
    assert side["n_joints"] == len(side["joint_names"])


def test_panda_text_block_does_not_add_a_label_line():
    """When the registry carries no labels, the text block gains no new line.

    Any reader that greps the ``get_features`` summary for the per-robot line
    reads exactly the string it read on main for panda / g1 / go2.
    """
    sim = Robot("panda")

    text = _text(sim, "panda")

    assert "joint_labels:" not in text
    # The per-robot summary line is still there, verbatim.
    assert "panda: 9 joints, 8 actuators" in text


# --------- so100: labels present (identity-ish but still carried) -------------


def test_so100_sidecar_carries_registry_labels_even_when_mjcf_names_match():
    """so100's upstream MJCF names joints semantically, but the registry still
    carries a ``joint_labels`` map (``Rotation → shoulder_pan``, ...). The
    sidecar surfaces it the same way as so101 so the two sibling arms read the
    same way from a reader's perspective.
    """
    sim = Robot("so100")

    side = _sidecar(sim, "so100")
    assert side["joint_labels"] == {
        "Rotation": "shoulder_pan",
        "Pitch": "shoulder_lift",
        "Elbow": "elbow_flex",
        "Wrist_Pitch": "wrist_flex",
        "Wrist_Roll": "wrist_roll",
        "Jaw": "gripper",
    }


# --------- whole-world listing also carries labels per robot ------------------


def test_whole_world_listing_carries_labels_per_robot():
    """A whole-world ``get_features()`` (no ``robot_name``) also enriches each
    robot's sidecar with its own labels, so a multi-robot scene reads the same
    shape as the robot-scoped call.
    """
    sim = Robot("so101")
    sim.add_robot("so100", position=[0.5, 0.0, 0.0])

    result = sim.get_features()
    sidecar = next(item["json"] for item in result["content"] if "json" in item)
    robots = sidecar["features"]["robots"]

    assert robots["so101"]["joint_labels"] == _SO101_LABELS
    assert robots["so100"]["joint_labels"]["Rotation"] == "shoulder_pan"
