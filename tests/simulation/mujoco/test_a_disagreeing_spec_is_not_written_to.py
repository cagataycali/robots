# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A durable scene mutation refuses a spec that disagrees with the compiled model.

``set_geom_properties`` and ``set_body_properties`` change the live model AND record
the change in the ``MjSpec`` the model is compiled from, because the next scene
mutation recompiles that spec over the model - a change written only to the model is
silently reverted while the caller has been told it took effect.

Recording it means locating the spec element a compiled id was built from, and the
compiler numbers elements in declaration order, so the id indexes the spec's element
list directly. That mapping is the whole risk: if the two representations no longer
agree, the element at a given index belongs to a DIFFERENT part of the machine, and
writing there moves one body's mass or one geom's shape onto another - a corruption
the caller cannot see, cannot undo, and did not ask for.

Six divergences, each caught by a different check and named by a different reason:

    a body missing before the target        it compiles to 3 bodies against the
                                            model's 4
    a body missing after the target         body id 3 is outside the scene spec's
                                            3 body(s)
    an element added (every id reset)       the geom at index 1 reports id -1
    ... the same, through the mass path     the body at index 1 reports id -1
    the target body renamed at its id       body 1 is named differently in a fresh
                                            compile
    the target's geom gone from the spec    the body declares no explicit inertial
                                            and owns no geom

The last three are what make a count check insufficient on its own: a spec can hold
the right NUMBER of elements and still not describe the same ones.

No public call leaves the two disagreeing - every scene mutation snapshots the spec
first and restores it when the recompile it precedes fails (see
``test_spec_snapshot_refusal``). The divergences below are therefore constructed on
the live spec directly, because what is under test is what these guards do when they
fire, and only a disagreeing spec can show that they refuse rather than write.

Each test asserts the whole cost of the refusal rather than only its status: every
geom size, every inertial row and every spec-side value in the scene is unchanged,
so a guard that reported an error AFTER writing somewhere would fail it. The control
test makes that meaningful by showing the identical calls succeed - and do change
those values - on a spec that still agrees.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

mj = pytest.importorskip("mujoco")

import numpy as np  # noqa: E402

from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# ``marker`` carries no geom and is declared BETWEEN the two geom-carrying bodies, so
# deleting it from the spec shifts the body numbering without touching the geom
# numbering - which is what lets one spec state show both sides of a missing body.
_SCENE = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1"/>
    <body name="crate" pos="0 0 1"><freejoint/>
      <geom name="crate_g" type="box" size="0.1 0.1 0.1" density="900"/>
    </body>
    <body name="marker" pos="0 0 2"><site name="marker_s" size="0.01"/></body>
    <body name="ball" pos="1 0 1"><freejoint/>
      <geom name="ball_g" type="sphere" size="0.08" density="700"/>
    </body>
  </worldbody>
</mujoco>
"""

_MARKER_INDEX = 2


@pytest.fixture
def sim():
    """A compiled scene whose spec still agrees with its model."""
    simulation = Simulation(tool_name="disagreeing_spec", mesh=False)
    assert simulation.create_world()["status"] == "success"
    assert simulation.replace_scene_mjcf(_SCENE)["status"] == "success"
    yield simulation
    simulation.cleanup()


def _spec(simulation) -> Any:
    return simulation._world._backend_state["spec"]


def _geom_id(simulation, name: str) -> int:
    return int(mj.mj_name2id(simulation._world._model, mj.mjtObj.mjOBJ_GEOM, name))


def _observable(simulation) -> dict[str, np.ndarray]:
    """Every value the two setters can write, across the whole scene.

    Read wholesale rather than per element: a refusal that wrote to the entity at
    another's index would leave the named one untouched, so only the whole scene
    can show that nothing moved.
    """
    model, spec = simulation._world._model, _spec(simulation)
    return {
        "model_geom_size": np.array(model.geom_size, dtype=float),
        "model_body_mass": np.array(model.body_mass, dtype=float),
        "model_body_ipos": np.array(model.body_ipos, dtype=float),
        "model_body_iquat": np.array(model.body_iquat, dtype=float),
        "model_body_inertia": np.array(model.body_inertia, dtype=float),
        "spec_geom_size": np.array([list(g.size) for g in spec.geoms], dtype=float),
        "spec_body_mass": np.array([float(b.mass) for b in spec.bodies], dtype=float),
        "spec_geom_density": np.array([float(g.density) for g in spec.geoms], dtype=float),
    }


def _assert_unchanged(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> None:
    for key, value in before.items():
        assert after[key] == pytest.approx(value, abs=1e-12), key


# The divergences, as (make it diverge, issue a durable change, the reason it names).
def _delete_marker(simulation) -> None:
    spec = _spec(simulation)
    spec.delete(spec.bodies[_MARKER_INDEX])


def _grow_a_body(simulation) -> None:
    extra = _spec(simulation).worldbody.add_body(name="extra")
    extra.add_site(name="extra_s", size=[0.01, 0.01, 0.01])


def _rename_the_crate(simulation) -> None:
    _spec(simulation).bodies[1].name = "crate_renamed"


def _delete_the_crates_geom(simulation) -> None:
    spec = _spec(simulation)
    spec.delete(spec.geoms[1])


def _resize_the_crate(simulation) -> dict[str, Any]:
    return simulation.set_geom_properties(geom_name="crate_g", size=[0.2, 0.2, 0.2])


def _resize_the_ball(simulation) -> dict[str, Any]:
    return simulation.set_geom_properties(geom_name="ball_g", size=[0.15])


def _set_the_crate_mass(simulation) -> dict[str, Any]:
    return simulation.set_body_properties(body_name="crate", mass=5.0)


_DIVERGENCES = [
    pytest.param(
        _delete_marker,
        _resize_the_crate,
        "it compiles to 3 bodies against the model's 4",
        id="a body missing before the target: the counts differ",
    ),
    pytest.param(
        _delete_marker,
        _resize_the_ball,
        "body id 3 is outside the scene spec's 3 body(s)",
        id="a body missing before the target: the id is past the end",
    ),
    pytest.param(
        _grow_a_body,
        _resize_the_crate,
        "the geom at index 1 reports id -1",
        id="an element added: the geom no longer knows its id",
    ),
    pytest.param(
        _grow_a_body,
        _set_the_crate_mass,
        "the body at index 1 reports id -1",
        id="an element added: the body no longer knows its id",
    ),
    pytest.param(
        _rename_the_crate,
        _resize_the_crate,
        "body 1 is named differently in a fresh compile",
        id="right count, different body: renamed at the same id",
    ),
    pytest.param(
        _delete_the_crates_geom,
        _set_the_crate_mass,
        "the body declares no explicit inertial and owns no geom",
        id="the element the change needs is gone: no geom to scale",
    ),
]


@pytest.mark.parametrize(("diverge", "mutate", "reason"), _DIVERGENCES)
def test_a_change_is_refused_and_the_disagreement_named(
    sim,
    diverge: Callable[[Any], None],
    mutate: Callable[[Any], dict[str, Any]],
    reason: str,
) -> None:
    """The reason has to name the disagreement: it is the only clue the caller gets."""
    diverge(sim)
    before = _observable(sim)

    result = mutate(sim)

    assert result["status"] == "error", result
    text = result["content"][0]["text"]
    # The setter names itself, so the reason reaches the caller attributed rather
    # than as a bare sentence from a helper it never called.
    assert text.startswith(("set_geom_properties: ", "set_body_properties: ")), text
    assert reason in text, text
    _assert_unchanged(before, _observable(sim))


def test_the_same_changes_are_applied_while_the_two_agree(sim) -> None:
    """The control: the refusals above are the disagreement, not the calls."""
    assert _resize_the_crate(sim)["status"] == "success"
    assert _resize_the_ball(sim)["status"] == "success"
    assert _set_the_crate_mass(sim)["status"] == "success"

    model = sim._world._model
    assert model.geom_size[_geom_id(sim, "crate_g")] == pytest.approx([0.2, 0.2, 0.2])
    assert float(model.geom_size[_geom_id(sim, "ball_g")][0]) == pytest.approx(0.15)
    assert float(model.body_mass[1]) == pytest.approx(5.0)


def test_the_fixture_diverges_in_the_way_each_reason_names(sim) -> None:
    """Each divergence has to BE one, or the refusals above prove nothing.

    Also pins the two spec behaviours the table rests on: deleting renumbers what
    remains (so the ids still agree and only the counts betray the difference),
    while adding invalidates every recorded id at once.
    """
    model = sim._world._model
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    assert names == ["world", "crate", "marker", "ball"]
    # The geomless body sits between the two targets, which is what makes one
    # deletion reach both a count mismatch and an out-of-range id.
    assert int(model.geom_bodyid[_geom_id(sim, "crate_g")]) < _MARKER_INDEX
    assert int(model.geom_bodyid[_geom_id(sim, "ball_g")]) > _MARKER_INDEX

    _delete_marker(sim)
    spec = _spec(sim)
    assert [(b.name, b.id) for b in spec.bodies] == [("world", 0), ("crate", 1), ("ball", 2)]
    assert len(spec.bodies) < model.nbody
    # Untouched, so the geom lookup still succeeds and the refusal is the body's.
    assert [(g.name, g.id) for g in spec.geoms] == [("ground", 0), ("crate_g", 1), ("ball_g", 2)]

    _delete_the_crates_geom(sim)
    # Deleting a geom renumbers only geoms, so the body lookup still succeeds and
    # the refusal is about what the body no longer owns.
    assert [b.id for b in spec.bodies] == [0, 1, 2]
    assert [g.name for g in spec.geoms] == ["ground", "ball_g"]
    assert list(spec.bodies[1].geoms) == []

    _grow_a_body(sim)
    # The world body is body 0 in every model and keeps its id; every element the
    # compiler could renumber loses the one it was carrying, whatever the spec
    # holds by now - which is why one added element invalidates the whole mapping.
    assert [g.id for g in spec.geoms] == [-1] * len(spec.geoms)
    assert [b.id for b in spec.bodies] == [0] + [-1] * (len(spec.bodies) - 1)


_NOTHING_TO_READ = [
    pytest.param(
        "_model",
        "the scene has no compiled model whose inertia could be re-derived",
        id="no compiled model",
    ),
    pytest.param("spec", "no live spec", id="no live spec"),
]


@pytest.mark.parametrize(("missing", "reason"), _NOTHING_TO_READ)
def test_the_refresh_needs_both_representations_to_read_either(sim, missing: str, reason: str) -> None:
    """Re-deriving an inertial row compares a fresh compile against the live model.

    Called with either side absent it says so and writes nothing, rather than
    reading through a ``None``. The setter reaches it only after resolving both,
    so this is the contract for a direct caller of the module function.
    """
    world = sim._world
    if missing == "_model":
        world._model = None
    else:
        world._backend_state.pop(missing)

    assert reason in str(scene_ops.refresh_body_inertial_from_geometry(world, 1))
