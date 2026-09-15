"""attach_bodies says whether the two bodies touch, and by how much they miss.

An agent running the README quickstart ("pick up the red cube") could not
reach the cube, closed the gripper centimetres away, welded the cube to the
jaw with ``attach_bodies`` and reported a successful pick. The object was
riding along 7 cm below the fingers. The tool now measures the closest
surface distance between the parent's subtree and the child, prints it when
they do not touch, and puts ``gap_m`` / ``touching`` in the json payload.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="t", mesh=False)
    s.create_world()
    # add_object sizes are full extents: carrier spans z 0.475..0.525.
    s.add_object("carrier", shape="box", size=[0.05, 0.05, 0.05], position=[0, 0, 0.5])
    yield s
    s.cleanup()


def _parts(result):
    text = next(c["text"] for c in result["content"] if "text" in c)
    payload = next(c["json"] for c in result["content"] if "json" in c)
    return text, payload


@pytest.mark.parametrize("mode", ["weld", "kinematic"])
def test_a_gap_is_named_in_centimetres(sim, mode):
    sim.add_object("cube", shape="box", size=[0.02, 0.02, 0.02], position=[0, 0, 0.57])
    result = sim.attach_bodies("carrier", "cube", mode=mode)
    assert result["status"] == "success", "attaching at a distance stays allowed - mounting is a real use"
    text, payload = _parts(result)
    assert "NOT touching" in text
    assert "3.5 cm" in text
    assert "not a pick" in text
    assert payload["touching"] is False
    assert payload["gap_m"] == pytest.approx(0.035, abs=1e-3)
    assert payload["mode"] == mode


# carrier spans z 0.475..0.525: a cube of half-extent 0.01 rests flush at 0.535
# and overlaps the carrier at 0.520.
@pytest.mark.parametrize("height_m", [0.535, 0.520])
def test_touching_bodies_say_so(sim, height_m):
    sim.add_object("cube", shape="box", size=[0.02, 0.02, 0.02], position=[0, 0, height_m])
    text, payload = _parts(sim.attach_bodies("carrier", "cube", mode="weld"))
    assert "The bodies are touching." in text
    assert "NOT touching" not in text
    assert payload["touching"] is True
    # Overlapping surfaces are touching, not a negative distance apart.
    assert payload["gap_m"] == 0.0


def test_robot_parent_measures_from_its_whole_subtree(sim):
    """A gripper's pads live on links below the body the caller names.

    ``so101/moving_jaw_so101_v1`` hangs below ``so101/gripper``, so a cube
    parked beside the jaw is nearer the jaw than anything the gripper body
    itself carries: naming the gripper has to answer like naming the jaw.
    """
    sim.add_robot("so101")
    sim.add_object("red_cube", shape="box", size=[0.02, 0.02, 0.02], position=[0.0394, -0.3809, 0.3271])
    text, from_parent = _parts(sim.attach_bodies("so101/gripper", "red_cube", mode="kinematic"))
    sim.detach_bodies("so101/gripper", "red_cube")
    _, from_link = _parts(sim.attach_bodies("so101/moving_jaw_so101_v1", "red_cube", mode="kinematic"))
    assert from_parent["gap_m"] == pytest.approx(from_link["gap_m"], abs=1e-9)
    assert from_parent["touching"] is False
    assert "floats rigidly" in text


@pytest.mark.parametrize(("offset_m", "gap_cm"), [(0.2, "16.5"), (2.0, "196.5"), (10.0, "996.5")])
def test_a_gap_wider_than_the_search_bound_is_still_the_real_distance(sim, offset_m, gap_cm):
    """A distance search bounded by a constant reports that constant.

    ``mj_geomDistance`` returns ``distmax`` unchanged when the true distance is
    larger, so a fixed bound of one metre would report a body 2 m away and one
    10 m away as the same "100.0 cm apart" - a number the caller cannot tell
    from a measurement. The bound is derived per pair, so the gap is real.
    """
    sim.add_object("cube", shape="box", size=[0.02, 0.02, 0.02], position=[offset_m, 0, 0.5])
    text, payload = _parts(sim.attach_bodies("carrier", "cube", mode="weld"))
    assert f"{gap_cm} cm apart" in text
    assert payload["gap_m"] == pytest.approx(float(gap_cm) / 100, abs=1e-3)


@pytest.mark.parametrize("break_measurement", ["absent", "answers_with_the_bound"])
def test_an_unmeasurable_gap_is_left_unknown(sim, monkeypatch, break_measurement):
    """No measurement means no sentence and null fields - never a guess."""
    if break_measurement == "absent":
        monkeypatch.delattr(mujoco, "mj_geomDistance")  # a MuJoCo build without it
    else:
        monkeypatch.setattr(mujoco, "mj_geomDistance", lambda m, d, g1, g2, distmax, fromto: distmax)
    sim.add_object("cube", shape="box", size=[0.02, 0.02, 0.02], position=[2.0, 0, 0.5])
    text, payload = _parts(sim.attach_bodies("carrier", "cube", mode="weld"))
    assert payload["gap_m"] is None
    assert payload["touching"] is None
    assert "touching" not in text
    assert "cm apart" not in text
