"""``contact_between`` answers for the names a scene lists, not only for raw geom names.

``add_object("cube")`` builds a geom named ``cube_geom``, and ``get_contacts``
reports that name. ``contact_between`` compared exact geom names, so
``contact_between("cube", "tray")`` stayed ``False`` while the two objects were
touching, with no warning, and a success clause written with the object names
scored 0%. Each side now matches a geom that IS the name or belongs to the body
of that name, through the same mapping ``grasped`` and
``body_on(require_contact=True)`` use.
"""

from __future__ import annotations

import importlib.util

import pytest

from strands_robots.simulation.predicates import make_predicate


class _Sim:
    def __init__(self, *pairs, active=True):
        self.contacts = [{"geom1": a, "geom2": b, "active": active} for a, b in pairs]

    def get_contacts(self):
        return {"status": "success", "content": [{"json": {"contacts": self.contacts}}]}


def _touching(sim, a, b) -> bool:
    return make_predicate("contact_between", geom_a=a, geom_b=b)(sim)


def test_object_names_match_the_geoms_add_object_builds():
    assert _touching(_Sim(("tray_geom", "cube_geom")), "cube", "tray")


def test_either_order_and_either_side_as_a_raw_geom_name():
    sim = _Sim(("cube_geom", "tray_geom"))
    assert _touching(sim, "tray", "cube")
    assert _touching(sim, "cube_geom", "tray")
    assert _touching(sim, "cube_geom", "tray_geom")


def test_a_robot_link_matches_its_unnamed_geoms():
    sim = _Sim(("so100/Fixed_Jaw/geom_31", "cube_geom"))
    assert _touching(sim, "so100/Fixed_Jaw", "cube")
    # a parent body does not claim a child link's geoms
    assert not _touching(sim, "so100", "cube")


def test_a_neighbouring_name_does_not_match():
    sim = _Sim(("cube_10_geom", "tray_geom"))
    assert not _touching(sim, "cube_1", "tray")


def test_both_sides_must_be_in_the_same_contact():
    sim = _Sim(("cube_geom", "ground"), ("tray_geom", "ground"))
    assert not _touching(sim, "cube", "tray")
    assert _touching(sim, "cube", "ground")


def test_an_inactive_proximity_record_is_still_not_contact():
    assert not _touching(_Sim(("tray_geom", "cube_geom"), active=False), "cube", "tray")


@pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="needs mujoco")
def test_a_cube_resting_on_a_tray_in_mujoco_is_in_contact_with_it():
    from strands_robots.simulation import create_simulation

    sim = create_simulation("mujoco")
    try:
        sim.create_world()
        sim.add_object("tray", shape="box", position=[0.3, 0.0, 0.02], size=[0.08, 0.08, 0.02])
        sim.add_object("cube", shape="box", position=[0.3, 0.0, 0.06], size=[0.015, 0.015, 0.015], mass=0.05)
        sim.add_object("ball", shape="sphere", position=[0.0, 0.3, 0.05], size=[0.02], mass=0.05)
        sim.step(200)
        assert _touching(sim, "cube", "tray")
        assert not _touching(sim, "cube", "ball")
        sim.move_object("cube", position=[0.3, 0.0, 0.5])
        sim.step(1)
        assert not _touching(sim, "cube", "tray")
    finally:
        sim.destroy()
