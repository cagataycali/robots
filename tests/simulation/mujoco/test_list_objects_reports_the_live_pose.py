"""``list_objects`` reports where an object *is*, not where it was *put*.

The scene record (:class:`~strands_robots.simulation.models.SimObject`) holds
the pose ``add_object`` / ``move_object`` was *asked* for. It is never written
by the physics, so reporting it described every manipuland by its last request:

* an object spawned above its rest height reported the spawn height for the rest
  of the session -- and a free-fall settle is the first thing a scene does, so
  the very first step makes the reading stale;
* an object the robot pushed reported no motion at all.

That is the number a caller grounds "did the object move" on -- the primary
success signal for a manipulation rollout -- so a stale placement is a silent
wrong answer, not a cosmetic one. The reported position now comes from
``mjData``.

The behaviour classes assert *physical* facts (a released object ends up near
the ground, below where it was released) rather than re-deriving the lookup the
implementation performs, so they grade the reported pose rather than restating
the patch. ``TestTheCommandedPoseIsStillReported`` and
``TestAStaticObjectStillReportsItsPlacement`` are the over-reach controls: they
pass on both trees, because the request-derived answer was already correct for
an object nothing has moved, and the fix must not perturb it.
"""

from __future__ import annotations

import re

import pytest

_ENTRY = re.compile(r"-\s+(?P<name>\S+):\s+(?P<shape>\S+)\s+at\s+\[(?P<pos>[^\]]*)\]")


@pytest.fixture
def sim():
    """A live MuJoCo world; each test gets its own so object names stay free."""
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    engine = Simulation(tool_name="devx_list_objects_live_pose", mesh=False)
    engine.create_world(ground_plane=True)
    try:
        yield engine
    finally:
        engine.cleanup(policy_stop_timeout=0.5)


def _reported(result: dict, name: str) -> list[float]:
    """The position ``list_objects`` publishes for ``name``."""
    assert result["status"] == "success", result
    text = result["content"][0]["text"]
    for match in _ENTRY.finditer(text):
        if match.group("name") == name:
            return [float(v) for v in match.group("pos").split(",")]
    raise AssertionError(f"{name!r} not listed in:\n{text}")


class TestASettledObjectReportsWhereItSettled:
    """The first physical event in any scene already made the old answer stale."""

    def test_a_released_object_is_not_reported_at_its_release_height(self, sim):
        release_z = 0.25
        half_extent = 0.03
        sim.add_object("brick", shape="box", size=[half_extent * 2] * 3, position=[0.0, 0.0, release_z])
        sim.step(600)

        reported = _reported(sim.list_objects(), "brick")

        # Non-vacuity: the object must really have fallen, or any answer passes.
        assert reported[2] < release_z - 0.1, f"premise: nothing fell, reported {reported}"
        assert reported[2] == pytest.approx(half_extent, abs=0.01), (
            f"a box resting on the ground plane sits at half its extent; reported {reported}"
        )


class TestAPushedObjectReportsThatItMoved:
    """The reading a manipulation rollout grounds success on."""

    def test_a_horizontal_shove_changes_the_reported_position(self, sim):
        sim.add_object("puck", shape="box", size=[0.05] * 3, position=[0.0, 0.0, 0.025])
        sim.step(60)
        before = _reported(sim.list_objects(), "puck")

        # A commanded lift, then a release: the object lands somewhere the
        # request never named, exactly as a gripper nudge leaves it.
        sim.move_object("puck", position=[0.14, 0.09, 0.30])
        sim.step(600)
        after = _reported(sim.list_objects(), "puck")

        assert after[2] < 0.30 - 0.1, f"premise: nothing fell, reported {after}"
        moved = max(abs(a - b) for a, b in zip(after, before, strict=True))
        assert moved > 0.05, f"a displaced object must not report its old pose: {before} -> {after}"


class TestTheCommandedPoseIsStillReported:
    """Over-reach control: the request was always right for an unmoved object."""

    def test_a_freshly_placed_object_reports_the_requested_position(self, sim):
        sim.add_object("token", shape="box", size=[0.04] * 3, position=[0.11, -0.07, 0.02])

        assert _reported(sim.list_objects(), "token") == pytest.approx([0.11, -0.07, 0.02], abs=1e-3)

    def test_move_object_is_reflected_before_anything_steps(self, sim):
        sim.add_object("token", shape="box", size=[0.04] * 3, position=[0.0, 0.0, 0.02])
        sim.move_object("token", position=[-0.06, 0.13, 0.02])

        assert _reported(sim.list_objects(), "token") == pytest.approx([-0.06, 0.13, 0.02], abs=1e-3)


class TestAStaticObjectStillReportsItsPlacement:
    """Over-reach control: a static body cannot move, so nothing may change."""

    def test_a_static_object_reports_its_placement_after_stepping(self, sim):
        sim.add_object("shelf", shape="box", size=[0.2, 0.2, 0.02], position=[0.0, 0.3, 0.15], is_static=True)
        sim.step(400)

        assert _reported(sim.list_objects(), "shelf") == pytest.approx([0.0, 0.3, 0.15], abs=1e-3)


class TestARecordWithNoBodyIsReportedNotPrinted:
    """The stale record is exactly what must not be printed when it is all there is."""

    def test_an_object_the_model_has_no_body_for_is_an_error(self, sim):
        from strands_robots.simulation.models import SimObject

        sim.add_object("real", shape="box", size=[0.04] * 3, position=[0.0, 0.0, 0.02])
        # A record whose body the compiled model does not carry. No caller is
        # known to produce this, so it is built directly -- the pin is that an
        # unresolvable name yields neither of the two silent wrong answers
        # available here: the record's placement, or the pose of whichever body
        # a ``-1`` index lands on.
        sim._world.objects["ghost"] = SimObject(name="ghost", shape="box", position=[9.0, 9.0, 9.0])

        result = sim.list_objects()

        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "ghost" in text
        assert "9.0" not in text, f"the record's placement must not be published: {text}"
