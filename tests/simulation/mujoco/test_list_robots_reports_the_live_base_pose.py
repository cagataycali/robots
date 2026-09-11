"""``list_robots_info`` reports where a robot is, not where it was asked to go.

The reported ``Position`` came from the scene record (``SimRobot.position``),
which holds the vector ``add_robot`` was *asked* for and is never written by the
physics. That is wrong in two independent ways.

It is wrong at ``t=0``, because ``position`` is the attach FRAME's translation
and MuJoCo COMPOSES it with the model's own authored root pose instead of
replacing it. 29 of the 51 single-root robots in the built-in registry therefore
never stood where the request named: a ``jvrc`` asked for ``z=0`` has its pelvis
at ``z=1.4``, a ``unitree_g1`` at ``z=0.793``, a ``unitree_go2`` at ``z=0.445``.
``add_robot`` already reports the measured placement, so the two calls
contradicted each other for the same robot in the same session.

And it stays wrong, because 32 of the 63 have a floating base. A robot that
walks, drives, flies or falls kept reporting its spawn pose for the rest of the
run, so the listing showed 0 mm of displacement for the motion that *is* the
success signal of a locomotion rollout - the reading a caller grounds "did the
robot get there" on.

These tests pin that the listed position is the MEASURED world position of the
robot's root body, both before anything steps and after the robot has moved, and
that it agrees with ``get_body_state`` and with ``add_robot``.

The controls carry as much weight: a robot whose root offset is zero - every
ground-bolted arm, 22 of the 51 - must read exactly as it read before, so the
change is scoped to the robots that were misreporting. A model with several
roots has no one base pose to measure, so it still reports the request, but
labelled as the request rather than implied to be a measurement.

Hermetic: inline MJCF written to ``tmp_path``, so no asset download. GL-free:
``mesh=False`` and no rendering.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

#: A floating-base model authored standing: its root declares ``z=0.5``, so
#: ``position`` composes with that, and the free joint lets it move afterwards.
#: This is the shape of every locomotion model in the registry (the Unitree Go2
#: base at ``z=0.445``, the JVRC pelvis at ``z=1.4``).
_STANDING_XML = """
<mujoco model="stander">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.05 0.04"/>
    </body>
  </worldbody>
</mujoco>
"""

#: A ground-bolted arm: root ``pos="0 0 0"`` and no free joint, so the requested
#: vector already IS the world position and stays that way. The control.
_BOLTED_XML = """
<mujoco model="bolted">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="box" size="0.04 0.04 0.02"/>
      <body name="link" pos="0 0 0.05">
        <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

#: Two independent roots, each with its own pose - the shape an ``aloha`` (two
#: arm bases) or an ``rby1`` (six) attaches. A set of roots has no one base pose.
_TWO_ROOT_XML = """
<mujoco model="pair">
  <compiler angle="radian"/>
  <worldbody>
    <body name="left" pos="-0.4 0 0.02">
      <geom type="box" size="0.04 0.04 0.02"/>
    </body>
    <body name="right" pos="0.4 0 0.02">
      <geom type="box" size="0.04 0.04 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


def _spawn(tmp_path, xml: str, label: str, position: list[float]) -> tuple[Simulation, dict]:
    """Add a robot from *xml* and return the sim plus ``add_robot``'s result."""
    model = tmp_path / f"{label}.xml"
    model.write_text(xml)
    sim = Simulation(tool_name=f"test_list_robots_live_pose_{label}", mesh=False)
    sim.create_world(gravity=[0, 0, -9.81])
    result = sim.add_robot(name=label, urdf_path=str(model), position=position)
    assert result["status"] == "success", result
    return sim, result


def _listed_placement(sim: Simulation, label: str) -> str:
    """The text ``list_robots_info`` reports after ``Position:`` for *label*."""
    result = sim.list_robots_info()
    assert result["status"] == "success", result
    blocks = [b for b in result["content"][0]["text"].split("  - ")[1:] if b.startswith(label)]
    assert len(blocks) == 1, f"premise: one block for {label!r}, got {blocks}"
    match = re.search(r"Position: (.*?), Joints:", blocks[0], re.DOTALL)
    assert match is not None, f"premise: a Position field, got {blocks[0]!r}"
    return match.group(1)


def _measured_position(sim: Simulation, body: str) -> list[float]:
    """The world position of *body*, read through the public state surface."""
    state = sim.get_body_state(body)
    assert state["status"] == "success", state
    return [float(v) for v in state["content"][1]["json"]["position"]]


class TestTheListedPositionIsTheMeasuredOne:
    def test_a_standing_robot_is_not_listed_at_the_height_it_was_asked_for(self, tmp_path) -> None:
        sim, _ = _spawn(tmp_path, _STANDING_XML, "stander", [0.0, 0.0, 0.4])
        try:
            base = _measured_position(sim, "stander/base")
            assert base[2] == pytest.approx(0.9), "premise: the model's own 0.5 root composes with the 0.4 request"
            assert "0.9" in _listed_placement(sim, "stander"), (
                "the listing reported the requested z=0.4 for a base standing at z=0.9, so the one number a "
                "caller has for 'where is this robot' named a place it is not"
            )
        finally:
            sim.cleanup()

    def test_a_robot_that_fell_is_not_still_listed_at_its_spawn_pose(self, tmp_path) -> None:
        sim, _ = _spawn(tmp_path, _STANDING_XML, "stander", [0.0, 0.0, 0.4])
        try:
            sim.step(400)
            base = _measured_position(sim, "stander/base")
            assert base[2] < 0.5, f"premise: the free base fell toward the floor, got z={base[2]}"
            listed = _listed_placement(sim, "stander")
            assert f"{round(base[2], 4)}" in listed, (
                f"the listing reported {listed!r} for a base the physics moved to z={base[2]:.4f}: a rollout "
                "reading displacement off this line sees 0 mm no matter what the robot did"
            )
        finally:
            sim.cleanup()

    def test_the_listing_and_add_robot_agree_in_the_same_session(self, tmp_path) -> None:
        # Both surfaces answer "where is this robot". add_robot already measured
        # it, so disagreeing left a caller no way to tell which one to believe.
        sim, added = _spawn(tmp_path, _STANDING_XML, "stander", [0.0, 0.0, 0.4])
        try:
            reported = [line for line in added["content"][0]["text"].splitlines() if line.startswith("Position:")]
            assert len(reported) == 1, f"premise: one Position line from add_robot, got {reported}"
            assert "0.9" in reported[0], f"premise: add_robot measures the placement: {reported[0]!r}"
            assert "0.9" in _listed_placement(sim, "stander")
        finally:
            sim.cleanup()


class TestTheUnaffectedListingsAreUnchanged:
    def test_a_bolted_arm_lists_exactly_what_was_requested(self, tmp_path) -> None:
        sim, _ = _spawn(tmp_path, _BOLTED_XML, "bolted", [0.0, 0.5, 0.0])
        try:
            sim.step(100)
            assert _measured_position(sim, "bolted/base") == pytest.approx([0.0, 0.5, 0.0])
            assert _listed_placement(sim, "bolted") == "[0.0, 0.5, 0.0]", (
                "an arm with a zero root offset and no free joint stands where it was put, so its line must "
                "read as it read before"
            )
        finally:
            sim.cleanup()

    def test_a_multi_root_model_says_its_number_is_the_request(self, tmp_path) -> None:
        # A set of roots has no one base pose, so there is nothing to measure.
        # Reporting the request is then the only answer available - but it has
        # to say so, or it reads as a measurement like every other line.
        sim, _ = _spawn(tmp_path, _TWO_ROOT_XML, "pair", [0.0, 0.0, 0.0])
        try:
            listed = _listed_placement(sim, "pair")
            assert listed.startswith("[0.0, 0.0, 0.0]"), f"the requested attach frame is still named: {listed!r}"
            assert "request" in listed, (
                f"a number that is the request rather than a measurement must say so: {listed!r}"
            )
        finally:
            sim.cleanup()
