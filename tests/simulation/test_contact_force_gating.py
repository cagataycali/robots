"""Regression tests: contact predicates require a load-bearing contact.

``mjData.contact`` lists every geom pair within ``margin``. A pair that falls
inside ``gap`` is reported with ``exclude != 0``, ``efc_address < 0`` and exactly
zero force - MuJoCo admits a contact to the solver only when
``dist < margin - gap``. Reporting geometry alone therefore made a physically
airborne body read as "touching":

* ``unitree_go2`` ships ``margin="0.001"``, so a foot 0.5 mm off the floor
  counted as ground contact.
* every ``contact_any`` / ``contact_between`` / ``grasped`` /
  ``body_on(require_contact=True)`` check could pass in mid-air.

Separately, ``_body_contact`` matched contacts by the geom-name prefix
``<body>_g``, which no real asset produces: the overwhelming majority of shipped
geoms are unnamed, so ``get_contacts`` synthesizes ``"<body>/geom_<id>"`` (a
slash), and a single-geom body often names its geom exactly like the body. Every
LIBERO ``(on A B)`` goal is gated on that check, so it failed while the object
physically rested on the target.

These tests pin both: a zero-force proximity record is NOT contact, and a real
resting contact between unnamed geoms IS.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation import predicates as P  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A cube held 25 mm clear of the floor. ``margin`` is the force-generation
# threshold and ``gap`` an ADDITIONAL detection buffer on top of it, so a narrow
# margin plus a wide gap makes MuJoCo report the pair (25 mm < margin + gap)
# while generating no force (25 mm > margin). Every listed contact then carries
# exclude=1 ("in gap"), efc_address=-1 and exactly zero force - which is what a
# downstream "is it touching?" check must not read as contact. Widening margin
# instead would make the contact load-bearing: the solver measures penetration
# against ``includemargin``, so the cube would hover on a real 3.7 N force.
_AIRBORNE_XML = """
<mujoco>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 .1"/>
    <body name="cube" pos="0 0 0.045">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>
      <geom name="cube_g0" type="box" size=".02 .02 .02" margin="0.001" gap="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

# A mug settling onto a plate. Both geoms are UNNAMED, so get_contacts
# synthesizes "<body>/geom_<id>" names.
_RESTING_XML = """
<mujoco>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 .1"/>
    <body name="plate_1" pos="0 0 0.01">
      <geom type="box" size=".08 .08 .01"/>
    </body>
    <body name="mug_1" pos="0 0 0.05">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>
      <geom type="box" size=".02 .02 .02"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def airborne(tmp_path):
    path = tmp_path / "airborne.xml"
    path.write_text(_AIRBORNE_XML)
    sim = Simulation(tool_name="contact_gate_airborne", mesh=False)
    sim.load_scene(scene_path=str(path))
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    yield sim
    sim.destroy()


@pytest.fixture
def resting(tmp_path):
    path = tmp_path / "resting.xml"
    path.write_text(_RESTING_XML)
    sim = Simulation(tool_name="contact_gate_resting", mesh=False)
    sim.load_scene(scene_path=str(path))
    for _ in range(1500):
        mujoco.mj_step(sim.mj_model, sim.mj_data)
    yield sim
    sim.destroy()


def test_get_contacts_reports_force_and_exclude(airborne) -> None:
    """The payload must expose what makes a contact real, not only geometry."""
    payload = P._extract_json(airborne.get_contacts())
    contacts = payload["contacts"]
    assert contacts, "MuJoCo should still list the in-margin pair"
    for c in contacts:
        assert "normal_force" in c and "exclude" in c
        assert "body1" in c and "body2" in c
        # The pair sits in the detection buffer but outside margin, so MuJoCo
        # keeps it out of the solver: exclude=1 ("in gap"), efc_address=-1 and
        # exactly zero force (measured on mujoco 3.10.0). Both are asserted -
        # the flag is what get_contacts has to expose, the force is what a
        # load-bearing check gates on.
        assert c["exclude"] != 0
        assert c["normal_force"] == pytest.approx(0.0)


def test_zero_force_contact_is_not_contact_any(airborne) -> None:
    assert P.PREDICATE_REGISTRY["contact_any"]()(airborne) is False


def test_zero_force_contact_is_not_contact_between(airborne) -> None:
    check = P.PREDICATE_REGISTRY["contact_between"](geom_a="ground", geom_b="cube_g0")
    assert check(airborne) is False


def test_real_resting_contact_is_detected(resting) -> None:
    assert P.PREDICATE_REGISTRY["contact_any"]()(resting) is True


def test_body_contact_matches_unnamed_geoms(resting) -> None:
    """Body-level match must work when geoms are unnamed (the common case)."""
    assert P._body_contact(resting, "mug_1", "plate_1") is True
    assert P._body_contact(resting, "plate_1", "mug_1") is True


def test_body_on_with_require_contact_succeeds(resting) -> None:
    """The gate every LIBERO ``(on A B)`` goal runs through."""
    check = P.PREDICATE_REGISTRY["body_on"](body_a="mug_1", body_b="plate_1", require_contact=True)
    assert check(resting) is True


def test_load_bearing_filter_tolerates_legacy_payload() -> None:
    """A backend that omits the new keys keeps the old geometry-only behaviour."""
    legacy = {"contacts": [{"geom1": "a", "geom2": "b", "dist": -0.001}]}
    assert len(P._load_bearing_contacts(legacy)) == 1
