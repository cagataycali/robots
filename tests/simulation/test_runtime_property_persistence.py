"""Regression tests: a runtime property change survives the next scene mutation.

``set_body_properties`` (mass/inertia) and ``set_geom_properties``
(color/friction/size) wrote ``model.*`` for immediate effect but never touched the
live ``MjSpec``. Every scene mutation recompiles from that spec - and the
incremental ``add_object`` path appends to it rather than rebuilding from
``world.objects`` - so the change was silently reverted:

    set_body_properties(mass=5.0)
    set_geom_properties(color=green, friction=[2.0, ...], size=[0.06]*3)
        -> mass 5.00, green, friction 2.0, half-extent 0.060     (correct)
    add_object(...)
        -> mass 0.20, red,   friction 1.0, half-extent 0.020     (reverted)

No error and no warning, so a domain-randomisation or task-setup step that tuned
an object's mass or friction quietly lost it as soon as anything else was added
to the scene - and the physics ran with the original values for the rest of the
episode.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_MASS = 5.0
_GREEN = [0.0, 1.0, 0.0, 1.0]
_FRICTION = [2.0, 0.5, 0.001]
_HALF_EXTENT = 0.06


@pytest.fixture
def sim():
    s = Simulation(tool_name="runtime_property_persistence", mesh=False)
    s.create_world()
    assert (
        s.add_object(name="cube", shape="box", size=[0.04] * 3, position=[0.2, 0, 0.3], mass=0.2, color=[1, 0, 0, 1])[
            "status"
        ]
        == "success"
    )
    assert s.set_body_properties(body_name="cube", mass=_MASS)["status"] == "success"
    assert (
        s.set_geom_properties(geom_name="cube_geom", color=_GREEN, friction=_FRICTION, size=[_HALF_EXTENT] * 3)[
            "status"
        ]
        == "success"
    )
    yield s
    s.destroy()


def _assert_properties_held(sim) -> None:
    model = sim.mj_model
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    assert float(model.body_mass[body]) == pytest.approx(_MASS)
    assert [float(v) for v in model.geom_rgba[geom]] == pytest.approx(_GREEN)
    assert [float(v) for v in model.geom_friction[geom]] == pytest.approx(_FRICTION)
    assert float(model.geom_size[geom][0]) == pytest.approx(_HALF_EXTENT)


def test_setters_take_effect_immediately(sim) -> None:
    _assert_properties_held(sim)


def test_spec_is_updated_not_just_the_model(sim) -> None:
    """The spec is the recompile source of truth, so it must carry the values.

    Mass lives on the GEOM, not the ``MjsBody``: ``add_object`` declares it there
    so the compiler derives the rotational inertia from the real shape instead of a
    hardcoded constant. With no ``explicitinertial`` block a write to ``body.mass``
    is silently dropped at compile time, which is why a runtime
    ``set_body_properties(mass=...)`` is mirrored onto the geom.
    """
    spec = sim._world._backend_state["spec"]
    assert float(spec.geom("cube_geom").mass) == pytest.approx(_MASS)
    assert [float(v) for v in spec.geom("cube_geom").rgba] == pytest.approx(_GREEN)


def test_survives_add_object(sim) -> None:
    assert sim.add_object(name="other", shape="box", size=[0.03] * 3, position=[1, 1, 0.05])["status"] == "success"
    _assert_properties_held(sim)


def test_survives_add_robot(sim) -> None:
    assert sim.add_robot(name="panda")["status"] == "success"
    _assert_properties_held(sim)


def test_survives_remove_object(sim) -> None:
    sim.add_object(name="other", shape="box", size=[0.03] * 3, position=[1, 1, 0.05])
    assert sim.remove_object(name="other")["status"] == "success"
    _assert_properties_held(sim)


def test_survives_add_camera(sim) -> None:
    assert sim.add_camera(name="cam", position=[1, 1, 1], target=[0, 0, 0])["status"] == "success"
    _assert_properties_held(sim)


def test_survives_reset(sim) -> None:
    assert sim.reset()["status"] == "success"
    _assert_properties_held(sim)


def test_heavier_body_actually_behaves_heavier_after_a_mutation(sim) -> None:
    """End-to-end: the retained mass must reach the solver, not just the array."""
    sim.add_object(name="other", shape="box", size=[0.03] * 3, position=[1, 1, 0.05])
    model = sim.mj_model
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    # mj_setConst-derived constants must reflect the retained mass too.
    assert float(model.body_subtreemass[body]) == pytest.approx(_MASS)


def test_unknown_geom_still_errors(sim) -> None:
    assert sim.set_geom_properties(geom_name="no_such_geom", color=[1, 1, 1, 1])["status"] == "error"


def test_survives_the_full_rebuild_path(sim) -> None:
    """``remove_robot`` rebuilds declaratively from ``world.objects``.

    Two rebuild paths exist and they read DIFFERENT sources: the incremental one
    recompiles the live spec, but ``eject_robot_from_scene`` runs a full
    ``SpecBuilder.build`` from ``world.objects``. Mirroring a runtime change onto
    the spec alone therefore still reverted on ``remove_robot`` - mass 5.0 -> 0.2,
    green -> red, friction 2.0 -> 1.0, half-extent 0.06 -> 0.02.
    """
    assert sim.add_robot(name="b", data_config="panda", position=[1.5, 0, 0])["status"] == "success"
    _assert_properties_held(sim)

    assert sim.remove_robot(name="b")["status"] == "success"
    _assert_properties_held(sim)


def test_sim_object_records_the_runtime_values(sim) -> None:
    """The declarative rebuild reads these fields, so they must be in step."""
    obj = sim._world.objects["cube"]
    assert obj.mass == pytest.approx(_MASS)
    assert list(obj.color) == pytest.approx(_GREEN)
    assert list(obj.friction) == pytest.approx(_FRICTION)
    # SimObject.size is the full-edge convention add_object accepts.
    assert list(obj.size) == pytest.approx([_HALF_EXTENT * 2] * 3)
