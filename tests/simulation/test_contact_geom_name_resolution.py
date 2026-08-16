"""Regression tests: body-level contact predicates resolve a body's geoms.

``get_contacts`` reports a contact as a pair of geom NAMES, and for a geom the
asset left unnamed it synthesizes ``"<body>/geom_<id>"``. Two body-level
predicates consume those names -- ``grasped`` and
``body_on(require_contact=True)``, the gate every LIBERO ``(on A B)`` goal runs
through -- and they used two different matchers:

* ``_body_contact`` matched only the ``<body>_g`` prefix inline, so a geom named
  exactly after its body did not match. ``grasped`` accepted that same name via
  ``_geom_belongs_to_body``, so the two predicates returned OPPOSITE answers for
  one contact list.
* Neither matcher recognised ``"<body>/geom_<id>"`` -- the form ``get_contacts``
  itself emits. A Panda scene has 81 unnamed geoms out of 82, so on real MJCF the
  payload producer and its consumers disagreed about the name format and every
  contact check read as "not touching".

These tests pin the single shared mapping: both matchers agree on every naming
convention, the synthesized name resolves, and a namespaced child's geom is not
claimed by its parent body.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.simulation.predicates import (
    _body_contact,
    _geom_belongs_to_body,
    make_predicate,
)

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Every geom-naming convention a supported scene source produces, plus the
# negatives that must stay unmatched.
_MATCHING = [
    ("mug", "a single-geom scene whose geom is named after the body"),
    ("mug_geom", "the strands add_object convention"),
    ("mug_g0", "the LIBERO / robosuite <body>_g<idx> convention"),
    ("mug_g11", "a double-digit LIBERO geom index"),
    ("mug/geom_2", "the name get_contacts synthesizes for an unnamed geom"),
    ("mug/geom_81", "a double-digit synthesized id"),
]
_NOT_MATCHING = [
    ("mug_collision", "an unrelated suffix"),
    ("mugx", "a longer body name"),
    ("plate_g0", "another body's geom"),
    ("mug/geom_", "the synthesized prefix with no id"),
    ("mug/geom_x", "a non-numeric suffix"),
    ("mug/link0/geom_1", "a namespaced CHILD body's geom"),
]


class _ContactPairSim:
    """Engine exposing exactly one contact between two given geom names."""

    def __init__(self, geom1: str, geom2: str) -> None:
        self._geom1 = geom1
        self._geom2 = geom2

    def get_contacts(self) -> dict[str, Any]:
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "contacts": [
                            {"geom1": self._geom1, "geom2": self._geom2, "dist": -1e-4, "pos": [0.0, 0.0, 0.0]}
                        ]
                    }
                }
            ],
        }


def _contact_pair(geom1: str, geom2: str) -> Any:
    """Return a duck-typed engine exposing one contact between two geom names.

    Returned as ``Any`` deliberately. The predicates probe for ``get_contacts``
    via ``getattr`` and fall back when it is absent - documented in the
    ``strands_robots.simulation.predicates`` module docstring - so a stub
    exposing only that method is a valid engine for them, even though it does
    not satisfy the ``SimEngine`` annotation.
    """
    return _ContactPairSim(geom1, geom2)


class TestSharedGeomNameMapping:
    """``_geom_belongs_to_body`` owns the body-to-geom name mapping."""

    @pytest.mark.parametrize("geom, why", _MATCHING)
    def test_supported_conventions_resolve(self, geom: str, why: str) -> None:
        assert _geom_belongs_to_body(geom, "mug") is True, why

    @pytest.mark.parametrize("geom, why", _NOT_MATCHING)
    def test_unrelated_names_do_not_resolve(self, geom: str, why: str) -> None:
        assert _geom_belongs_to_body(geom, "mug") is False, why

    def test_index_boundary_keeps_distinct_bodies_apart(self) -> None:
        """``cube_1`` must not claim ``cube_10``'s geoms."""
        assert _geom_belongs_to_body("cube_10_g0", "cube_1") is False

    def test_a_parent_body_does_not_claim_a_namespaced_childs_geom(self) -> None:
        """The synthesized form is matched exactly, not as a ``<body>/`` prefix.

        ``add_robot`` namespaces bodies, so a Panda's links are ``panda/link0``
        and their unnamed geoms are reported as ``panda/link0/geom_<id>``. A
        broad ``panda/`` prefix would let the body ``panda`` claim them.
        """
        assert _geom_belongs_to_body("panda/link0/geom_1", "panda/link0") is True
        assert _geom_belongs_to_body("panda/link0/geom_1", "panda") is False


class TestBodyLevelPredicatesAgree:
    """The two body-level matchers cannot disagree about one contact list."""

    @pytest.mark.parametrize("geom, why", _MATCHING + _NOT_MATCHING)
    def test_body_contact_matches_the_shared_mapping(self, geom: str, why: str) -> None:
        expected = _geom_belongs_to_body(geom, "mug")
        sim = _contact_pair(geom, "plate_g0")
        assert _body_contact(sim, "mug", "plate") is expected, why

    @pytest.mark.parametrize("geom, why", _MATCHING)
    def test_body_contact_and_grasped_agree(self, geom: str, why: str) -> None:
        """One contact list, one verdict -- whichever predicate reads it."""
        sim = _contact_pair(geom, "gripper_pad")
        grasped = make_predicate("grasped", body="mug", gripper_prefix="gripper")
        assert _body_contact(sim, "mug", "gripper_pad") is True, why
        assert grasped(sim) is True, why

    def test_contact_is_direction_agnostic(self) -> None:
        """A pair reported (b, a) counts the same as (a, b)."""
        assert _body_contact(_contact_pair("plate/geom_1", "mug/geom_2"), "mug", "plate") is True
        assert _body_contact(_contact_pair("mug/geom_2", "plate/geom_1"), "mug", "plate") is True


# --- live MuJoCo scenes ------------------------------------------------------
#
# A cube resting on a plate under gravity. The two variants differ ONLY in
# whether the geoms carry names, which is what decides the reported name format.

_SCENE = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 .1"/>
    <body name="plate" pos="0 0 0.02">
      <geom {plate_name}type="box" size=".12 .12 .02"/>
    </body>
    <body name="mug" pos="0 0 0.10">
      <freejoint/>
      <geom {mug_name}type="box" size=".03 .03 .03"/>
    </body>
  </worldbody>
</mujoco>
"""


def _settled(xml: str):
    sim = Simulation(tool_name="contact_names", mesh=False)
    sim.create_world()
    sim.replace_scene_mjcf(xml)
    sim.step(400)
    return sim


@pytest.fixture
def unnamed_geoms():
    """A resting contact between geoms the asset left unnamed."""
    sim = _settled(_SCENE.format(plate_name="", mug_name=""))
    yield sim
    sim.cleanup()


@pytest.fixture
def geoms_named_after_bodies():
    """A resting contact where each body's single geom carries the body's name."""
    sim = _settled(_SCENE.format(plate_name='name="plate" ', mug_name='name="mug" '))
    yield sim
    sim.cleanup()


def _reported_pairs(sim) -> list[tuple[str, str]]:
    result = sim.get_contacts()
    assert result["status"] == "success", result
    payload = next(c["json"] for c in result["content"] if "json" in c)
    return [(c["geom1"], c["geom2"]) for c in payload["contacts"]]


class TestRestingContactIsDetected:
    """A body physically resting on another must read as touching."""

    @pytest.mark.parametrize("fixture_name", ["unnamed_geoms", "geoms_named_after_bodies"])
    def test_body_on_require_contact_succeeds(self, fixture_name: str, request) -> None:
        """The gate every LIBERO ``(on A B)`` goal is compiled with."""
        sim = request.getfixturevalue(fixture_name)
        # Premise: the scene really is in contact, so a False verdict below
        # would be a false negative rather than an honest miss.
        pairs = _reported_pairs(sim)
        assert pairs, "fixture reported no contacts - nothing to resolve"

        pred = make_predicate("body_on", body_a="mug", body_b="plate", require_contact=True)
        assert pred(sim) is True

    @pytest.mark.parametrize("fixture_name", ["unnamed_geoms", "geoms_named_after_bodies"])
    def test_every_reported_geom_resolves_to_its_body(self, fixture_name: str, request) -> None:
        """The payload producer and its consumers agree on the name format.

        Derived from the compiled model rather than from hand-written names, so a
        future change to how ``get_contacts`` names a geom fails here.
        """
        sim = request.getfixturevalue(fixture_name)
        pairs = _reported_pairs(sim)
        assert pairs, "fixture reported no contacts - nothing to resolve"
        for geom1, geom2 in pairs:
            assert _geom_belongs_to_body(geom1, "plate"), geom1
            assert _geom_belongs_to_body(geom2, "mug"), geom2

    def test_grasped_fires_on_unnamed_geoms(self, unnamed_geoms) -> None:
        """``grasped`` treats the other side as a geom-name prefix set."""
        pred = make_predicate("grasped", body="mug", gripper_prefix="plate")
        assert pred(unnamed_geoms) is True

    def test_an_unrelated_body_is_not_in_contact(self, unnamed_geoms) -> None:
        """The fix must not make every body read as touching every other."""
        assert _body_contact(unnamed_geoms, "mug", "ground") is False
