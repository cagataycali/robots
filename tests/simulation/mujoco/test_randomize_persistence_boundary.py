"""``randomize()`` survives a reset but not a scene mutation, and says so.

Every axis writes the compiled ``model``, so a :meth:`reset` and a plain
``step`` both keep the perturbation -- which is the whole reason the axis
reaches a rollout, because every rollout entry point resets before an episode's
first step. But the compiled model is *derived* state: every scene mutation
recompiles the scene spec over it, and the spec carries the authored values
deliberately, because the lighting and position axes measure their bounded
offsets from that reference. So a scene mutation restores the authored scene and
undoes every axis.

``randomize()``'s docstring claimed the opposite -- "Every axis persists for the
rest of the scene's life" -- and named no operation that ends it, while the one
undo it did name (``recompile the scene``) is not an action the tool surface
publishes at all. Seven published scene actions perform that undo incidentally,
and the sequence a per-episode loop reaches for first is the one that loses
everything: :meth:`~strands_robots.simulation.benchmark.BenchmarkProtocol.on_episode_start`
invites "per-episode randomization, goal sampling, or procedural scene
generation", and randomizing before the episode's objects are added leaves the
policy's first observation looking at an unrandomized scene with both calls
having returned ``status="success"``.

These tests measure the boundary from the running engine and then require every
surface that makes a persistence claim to name it, so the prose cannot drift
from the behaviour again. The behaviour itself is unchanged and is asserted
here as the control:
:mod:`tests.simulation.mujoco.test_randomize_positions_persistence` already
pins one axis of it (``test_a_recompile_still_undoes_the_position_axis``); this
generalises the same rule to every axis and every published route.

The same rule one level over: a surface may not promise an axis a *quantity*
the axis does not sample either. The guide described ``randomize_physics`` as
scaling "mass, friction and joint damping" in two places, while the axis writes
``geom_friction`` and ``body_mass`` only, and no ``damping_range`` exists in the
signature to ask for more - so a reader closed the sim2real gap on the one
dynamics parameter a position-controlled arm needs most, and nothing said the
axis had not moved it.
"""

from __future__ import annotations

import functools
import pathlib
import re
from typing import Any

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import _PUBLISHED_ACTIONS, Simulation  # noqa: E402

_DOCS = pathlib.Path(__file__).resolve().parents[3] / "docs" / "simulation" / "domain-randomization.md"

#: Model arrays each axis writes, grouped by the axis that owns them. A
#: targeted setter (``set_geom_properties`` / ``set_body_properties``) moves the
#: arrays of at most one axis; a scene mutation restores them all, so counting
#: distinct axes separates the two without a magic array count.
_AXIS_ARRAYS: dict[str, tuple[str, ...]] = {
    "colour": ("geom_rgba", "mat_rgba"),
    "friction": ("geom_friction",),
    "mass": ("body_mass", "body_inertia"),
    "position": ("qpos0",),
    "lighting": ("light_pos", "light_diffuse"),
}
_ARRAYS = tuple(name for arrays in _AXIS_ARRAYS.values() for name in arrays)

#: An operation that loses this many distinct axes has restored the authored
#: scene rather than changed one entity. Measured separation is wide: a targeted
#: setter loses one axis, a scene mutation loses four or five.
_WHOLESALE_AXES = 3

_ROBOT_MJCF = """<mujoco model="probe">
  <worldbody>
    <body name="link" pos="0 0 0.1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="link_geom" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.1"/>
    </body>
  </worldbody>
  <actuator><position name="act" joint="hinge" kp="10"/></actuator>
</mujoco>
"""


def _ok(result: dict[str, Any], what: str) -> dict[str, Any]:
    """Return ``result``, raising when the call that built the scene refused.

    A ``raise`` rather than an ``assert`` because these calls *are* the premise:
    under ``python -O`` a bare assert would drop the scene construction along
    with the check, and every measurement below would run against an empty world.
    """
    if result["status"] != "success":
        raise AssertionError(f"{what} refused: {result}")
    return result


def _arrays(sim: Simulation) -> dict[str, np.ndarray]:
    assert sim._world is not None and sim._world._model is not None
    model = sim._world._model
    return {name: np.asarray(getattr(model, name), dtype=np.float64).copy() for name in _ARRAYS}


def _lost_axes(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> frozenset[str]:
    """Axes whose arrays no longer hold the randomized values."""
    lost = set()
    for axis, names in _AXIS_ARRAYS.items():
        for name in names:
            a, b = before[name], after[name]
            n = min(len(a), len(b))
            if not np.allclose(a[:n], b[:n]):
                lost.add(axis)
                break
    return frozenset(lost)


def _randomized_sim(tmp_path: pathlib.Path) -> Simulation:
    """A scene with every axis randomized, plus the entities the ops need."""
    sim = Simulation(tool_name="test_randomize_persistence_boundary", mesh=False)
    _ok(sim.create_world(gravity=[0, 0, -9.81]), "create_world")
    _ok(
        sim.add_object(name="crate", shape="box", position=[0.3, 0.0, 0.05], size=[0.03] * 3, mass=0.2),
        "add_object(crate)",
    )
    _ok(sim.add_camera(name="cam0", position=[0.4, -0.4, 0.3], target=[0, 0, 0.1]), "add_camera(cam0)")
    urdf = tmp_path / "probe.xml"
    urdf.write_text(_ROBOT_MJCF)
    _ok(sim.add_robot(name="probe", urdf_path=str(urdf)), "add_robot(probe)")
    _ok(
        sim.randomize(
            randomize_colors=True,
            randomize_lighting=True,
            randomize_physics=True,
            randomize_positions=True,
            position_noise=0.02,
            seed=7,
        ),
        "randomize",
    )
    return sim


def _operations(tmp_path: pathlib.Path) -> dict[str, Any]:
    """Published operations to grade, keyed by the action name they publish."""
    marker = tmp_path / "second.xml"
    marker.write_text(_ROBOT_MJCF.replace('model="probe"', 'model="probe2"'))
    return {
        "reset": lambda s: s.reset(),
        "step": lambda s: s.step(n_steps=10),
        "set_geom_properties": lambda s: s.set_geom_properties(geom_name="crate_geom", color=[0.1, 0.2, 0.3, 1.0]),
        "set_body_properties": lambda s: s.set_body_properties(body_name="crate", mass=0.6),
        "add_object": lambda s: s.add_object(name="extra", shape="sphere", position=[0.4, 0.1, 0.05], size=[0.02]),
        "remove_object": lambda s: s.remove_object(name="crate"),
        "add_camera": lambda s: s.add_camera(name="cam1", position=[0.5, -0.5, 0.4], target=[0, 0, 0.1]),
        "remove_camera": lambda s: s.remove_camera(name="cam0"),
        "add_robot": lambda s: s.add_robot(name="probe2", urdf_path=str(marker)),
        "remove_robot": lambda s: s.remove_robot(name="probe"),
        "patch_scene_mjcf": lambda s: s.patch_scene_mjcf(
            ops=[{"op": "add_site", "name": "mk", "pos": [0.1, 0, 0.1], "size": [0.01] * 3}]
        ),
    }


@functools.lru_cache(maxsize=1)
def _measured(tmp: str) -> dict[str, frozenset[str]]:
    """Axes each published operation loses, measured from the running engine."""
    tmp_path = pathlib.Path(tmp)
    out: dict[str, frozenset[str]] = {}
    for name, op in _operations(tmp_path).items():
        sim = _randomized_sim(tmp_path)
        try:
            before = _arrays(sim)
            _ok(op(sim), name)
            out[name] = _lost_axes(before, _arrays(sim))
        finally:
            sim.cleanup()
    return out


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> dict[str, frozenset[str]]:
    return _measured(str(tmp_path_factory.mktemp("boundary")))


def _wholesale(measured: dict[str, frozenset[str]]) -> frozenset[str]:
    return frozenset(op for op, lost in measured.items() if len(lost) >= _WHOLESALE_AXES)


def _named_operations(text: str) -> frozenset[str]:
    """Operation names the prose names, in backticks or as a Sphinx role.

    Deliberately permissive about *where* a name appears: it asks only that the
    surface names the operation somewhere, so a surface that names one in an
    unrelated role still counts. That can under-report a surface, never
    over-report one -- which is the direction a grader over prose should err in.
    """
    tokens = set(re.findall(r"``([a-z_]+)``", text))
    tokens |= set(re.findall(r":meth:`~?[\w.]*?([a-z_]+)`", text))
    tokens |= set(re.findall(r"`([a-z_]+)(?:\(\.\.\.\))?`", text))
    return frozenset(tokens)


def _claims_persistence(text: str) -> bool:
    """Whether the prose tells a reader how long a randomization lasts."""
    flat = " ".join(text.split()).lower()
    return ("persist" in flat or "survive" in flat) and "reset" in flat


# --------------------------------------------------------------------------
# The boundary, measured from the engine. Unchanged by this branch.
# --------------------------------------------------------------------------


def test_a_reset_and_a_step_keep_every_axis(measured):
    """The half the docstring's reasoning depends on: a rollout resets first."""
    for op in ("reset", "step"):
        assert measured[op] == frozenset(), f"{op} lost {sorted(measured[op])}"


def test_a_targeted_setter_moves_at_most_one_axis(measured):
    """``set_geom_properties`` / ``set_body_properties`` change one entity.

    They are the reason the wholesale test counts axes rather than arrays: a
    mass change legitimately moves ``body_mass`` and ``body_inertia`` together.
    """
    for op in ("set_geom_properties", "set_body_properties"):
        assert len(measured[op]) <= 1, f"{op} lost {sorted(measured[op])}"


@pytest.mark.parametrize(
    "op",
    [
        "add_object",
        "remove_object",
        "add_camera",
        "remove_camera",
        "add_robot",
        "remove_robot",
        "patch_scene_mjcf",
    ],
)
def test_a_scene_mutation_restores_the_authored_scene(measured, op):
    """Every scene mutation recompiles the spec, undoing every axis."""
    assert len(measured[op]) >= _WHOLESALE_AXES, f"{op} lost only {sorted(measured[op])}"


def test_randomizing_before_a_scene_mutation_reaches_no_observation(tmp_path):
    """The sequence a per-episode loop reaches for first, and both calls succeed.

    This is what the docstring has to warn about: the caller is told the axis
    persists, and the ordering that reads naturally (randomize the scene, then
    add this episode's objects) leaves nothing randomized.
    """
    sim = Simulation(tool_name="test_randomize_persistence_boundary_order", mesh=False)
    try:
        _ok(sim.create_world(gravity=[0, 0, -9.81]), "create_world")
        _ok(
            sim.add_object(name="target", shape="box", position=[0.3, 0, 0.05], size=[0.03] * 3, mass=0.2),
            "add_object(target)",
        )
        authored = _arrays(sim)
        _ok(sim.randomize(randomize_colors=True, randomize_physics=True, seed=3), "randomize")
        assert _lost_axes(authored, _arrays(sim)), "premise: the randomization must have changed something"
        _ok(sim.add_object(name="distractor", shape="sphere", position=[0.4, 0.1, 0.04], size=[0.02]), "add_object")
        assert _lost_axes(authored, _arrays(sim)) == frozenset(), (
            "a scene mutation after randomize() left the authored scene, and both calls reported success"
        )
    finally:
        sim.cleanup()


# --------------------------------------------------------------------------
# Every surface that makes the persistence claim names the boundary.
# --------------------------------------------------------------------------


def test_the_randomize_docstring_names_what_ends_the_persistence(measured):
    """A claim about how long an axis lasts has to name what ends it."""
    doc = Simulation.randomize.__doc__ or ""
    assert _claims_persistence(doc), "premise: the docstring still discusses how long an axis lasts"
    missing = sorted(_wholesale(measured) - _named_operations(doc))
    assert not missing, (
        "randomize()'s docstring tells the caller how long a randomization lasts but never names "
        f"these operations, each measured to restore the authored scene: {missing}"
    )


def test_the_documentation_page_names_what_ends_the_persistence(measured):
    """The guide makes the same claim and needs the same boundary."""
    page = _DOCS.read_text(encoding="utf-8")
    assert _claims_persistence(page), f"premise: {_DOCS.name} still discusses how long an axis lasts"
    missing = sorted(_wholesale(measured) - _named_operations(page))
    assert not missing, f"{_DOCS.name} makes the persistence claim without naming these undo routes: {missing}"


def test_no_surface_offers_a_recompile_action_that_does_not_exist(measured):
    """The undo has to be reachable.

    The docstring used to say "recompile the scene to undo", and no published
    action is named ``recompile``: the seven measured routes are the undo.
    """
    published = set(_PUBLISHED_ACTIONS)
    assert "recompile" not in published, "premise: there is still no recompile action"
    for surface, text in (("randomize.__doc__", Simulation.randomize.__doc__ or ""), (_DOCS.name, _DOCS.read_text())):
        assert "recompile the scene to undo" not in text, f"{surface} offers an action the surface does not publish"
    assert _wholesale(measured) <= published, "every measured undo route must be a published action"


def test_a_claim_without_a_boundary_is_reported(measured):
    """Non-vacuity, both directions."""
    ops = sorted(_wholesale(measured))
    assert ops, "premise: at least one operation was measured to restore the authored scene"
    bare = "Every axis persists for the rest of the scene's life, including across reset()."
    assert _claims_persistence(bare)
    assert sorted(_wholesale(measured) - _named_operations(bare)) == ops
    qualified = bare + " A scene mutation undoes it: " + ", ".join(f"``{op}``" for op in ops) + "."
    assert not (_wholesale(measured) - _named_operations(qualified))
    # Prose that makes no claim is not graded at all.
    assert not _claims_persistence("Each flag is opt-in per-axis.")


# --------------------------------------------------------------------------
# The physics axis samples friction and mass, and no surface promises more.
# --------------------------------------------------------------------------

#: Dynamics quantities a reader could take ``randomize_physics`` to sample,
#: each with the model array that would carry it. Measured below: the axis
#: writes ``geom_friction`` and ``body_mass`` / ``body_inertia`` and leaves
#: every array here byte-identical -- and ``randomize()``'s signature has no
#: range parameter for any of them, so a surface naming one promises an axis
#: the tree does not have. Actuator damping is the one that matters most: it is
#: the first parameter a sim2real reader expects a physics axis to cover.
_UNSAMPLED_DYNAMICS: dict[str, str] = {
    "armature": "dof_armature",
    "damping": "dof_damping",
    "frictionloss": "dof_frictionloss",
    "stiffness": "jnt_stiffness",
}

#: A joint that authors every quantity above with a non-zero value, so "the
#: axis left it alone" is distinguishable from "the axis scaled a zero".
_DYNAMICS_MJCF = """<mujoco model="dynamics">
  <worldbody>
    <body name="link" pos="0 0 0.1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.5" stiffness="1.5"
             armature="0.01" frictionloss="0.02"/>
      <geom name="link_geom" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.1"/>
    </body>
  </worldbody>
  <actuator><position name="act" joint="hinge" kp="10"/></actuator>
</mujoco>
"""


@functools.lru_cache(maxsize=1)
def _physics_axis_deltas(tmp: str) -> dict[str, float]:
    """``max|delta|`` per model array after the physics axis runs alone."""
    urdf = pathlib.Path(tmp) / "dynamics.xml"
    urdf.write_text(_DYNAMICS_MJCF)
    names = ("geom_friction", "body_mass", *sorted(_UNSAMPLED_DYNAMICS.values()))
    sim = Simulation(tool_name="test_randomize_physics_axis_scope", mesh=False)
    try:
        _ok(sim.create_world(gravity=[0, 0, -9.81]), "create_world")
        _ok(sim.add_robot(name="dyn", urdf_path=str(urdf)), "add_robot(dyn)")
        assert sim._world is not None and sim._world._model is not None
        model = sim._world._model
        read = lambda: {n: np.asarray(getattr(model, n), dtype=np.float64).copy() for n in names}  # noqa: E731
        before = read()
        _ok(
            sim.randomize(randomize_colors=False, randomize_lighting=False, randomize_physics=True, seed=5), "randomize"
        )
        after = read()
    finally:
        sim.cleanup()
    return {n: float(np.abs(after[n] - before[n]).max()) for n in names}


@pytest.fixture(scope="module")
def physics_axis_deltas(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    return _physics_axis_deltas(str(tmp_path_factory.mktemp("physics_axis")))


def test_the_physics_axis_writes_friction_and_mass(physics_axis_deltas):
    """Premise for the scope test below: the axis did sample something."""
    for name in ("geom_friction", "body_mass"):
        assert physics_axis_deltas[name] > 0.0, f"{name} unchanged: {physics_axis_deltas}"


@pytest.mark.parametrize(("quantity", "array"), sorted(_UNSAMPLED_DYNAMICS.items()))
def test_the_physics_axis_leaves_every_other_dynamics_array_untouched(physics_axis_deltas, quantity, array):
    """Each quantity is authored non-zero, so an untouched array is a real no-op."""
    assert physics_axis_deltas[array] == 0.0, (
        f"randomize_physics moved {array} by {physics_axis_deltas[array]}, so the axis does sample {quantity}"
    )


def _physics_axis_claims(text: str) -> list[str]:
    """Lines describing the physics axis, where a named quantity is a promise."""
    markers = ("randomize_physics", "mass_range", "friction_range")
    return [line for line in text.splitlines() if any(marker in line for marker in markers)]


def _promised_but_unsampled(text: str) -> frozenset[str]:
    lines = " || ".join(_physics_axis_claims(text)).lower()
    return frozenset(q for q in _UNSAMPLED_DYNAMICS if q in lines)


@pytest.mark.parametrize(
    "surface",
    [
        "docs/simulation/domain-randomization.md",
        "examples/12_domain_randomization.py",
        "strands_robots/simulation/mujoco/randomization.py",
        "strands_robots/simulation/newton/randomization.py",
    ],
)
def test_no_surface_promises_a_dynamics_quantity_the_physics_axis_does_not_sample(physics_axis_deltas, surface):
    """A quantity named beside the axis is read as an axis the axis covers."""
    path = pathlib.Path(__file__).resolve().parents[3] / surface
    text = path.read_text(encoding="utf-8")
    assert _physics_axis_claims(text), f"premise: {surface} still describes the physics axis"
    promised = sorted(_promised_but_unsampled(text))
    assert not promised, (
        f"{surface} describes randomize_physics as sampling {promised}, each measured to leave its "
        f"model array byte-identical: { {_UNSAMPLED_DYNAMICS[q]: physics_axis_deltas[_UNSAMPLED_DYNAMICS[q]] for q in promised} }"
    )


def test_a_surface_promising_an_unsampled_quantity_is_reported():
    """Non-vacuity, both directions."""
    assert _promised_but_unsampled("randomize_physics scales mass, friction and joint damping") == {"damping"}
    assert _promised_but_unsampled("| `randomize_physics` | mass (mult), friction (scale) |") == frozenset()
    # A quantity named away from the axis is not a claim about the axis.
    assert _promised_but_unsampled("MuJoCo reads joint damping from the MJCF.") == frozenset()
