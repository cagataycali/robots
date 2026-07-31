"""The MuJoCo runtime writers refuse a boolean instead of writing it as 1.0/0.0.

``bool`` is an ``int`` subclass and ``numpy.bool_`` coerces identically, so a
``try: float(value)`` + ``math.isfinite`` coercion admits both and installs a
silent ``1.0`` -- a 1 radian joint target, a 1 N force component, a fully
saturated colour channel, a 1 m/s^2 gravity axis -- under ``status="success"``.

Four hand-rolled coercions in the MuJoCo path shared that shape while every
other numeric surface in the package refused a boolean and said why:
:func:`strands_robots.utils.finite_vector_error` (``add_object`` / ``move_object``),
``send_action``, the teleop wire validator, and the agent-tool router -- which
refuses a bool component of ``apply_force``'s own ``force`` / ``torque`` /
``point``, so the same method answered differently depending on the entry point.

The tests below pin both directions: every boolean spelling is refused at every
affected surface, and every value those surfaces accepted before (Python and
NumPy reals, integers, numeric strings, NumPy arrays) is still accepted. A
structural test keeps a coercion added later from reopening the hole.
"""

import ast
import inspect
import pathlib
import re

import numpy as np
import pytest

from strands_robots.simulation import Simulation
from strands_robots.utils import is_boolean

# Two hinges, explicit actuators, no downloaded asset -- ``angle="radian"`` is
# required because MJCF defaults to degrees, which would silently shrink the
# joint ranges by a factor of ~57.
_ARM_XML = """<mujoco model="probe_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" mass="1"/>
      <body name="link" pos="0 0 0.2">
        <joint name="pan" type="hinge" axis="0 0 1" damping="1" range="-3 3" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.025" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="pan_act" joint="pan" kp="20" ctrlrange="-3 3"/></actuator>
</mujoco>"""

# Every spelling of "boolean" that reaches these writers in practice: a literal
# flag, a NumPy boolean from a comparison (``gripper > 0.5``), and the 0-d array
# an indexed comparison yields. ``numpy.bool_`` is not a ``bool`` subclass, so
# an ``isinstance(value, bool)`` check alone misses two of the five.
BOOLEANS = [
    pytest.param(True, id="python-True"),
    pytest.param(False, id="python-False"),
    pytest.param(np.True_, id="numpy-True_"),
    pytest.param(np.bool_(False), id="numpy-bool_-False"),
    pytest.param(np.array(True), id="numpy-0d-array-True"),
]


@pytest.fixture
def sim(tmp_path):
    # Intentionally un-annotated: annotating this as the engine type makes mypy
    # read ``sim._world`` as ``SimWorld | None`` at every model probe below.
    arm = tmp_path / "arm.xml"
    arm.write_text(_ARM_XML)
    engine = Simulation(backend="mujoco", mesh=False)
    engine.create_world()
    engine.add_robot(name="arm", urdf_path=str(arm))
    engine.add_object(name="cube", shape="box", size=[0.1, 0.1, 0.1], position=[0.4, 0.0, 0.3])
    yield engine
    engine.cleanup()


def _text(result):
    return "\n".join(c["text"] for c in result.get("content", []) if "text" in c)


# Each entry is (surface label, call taking the offending value). Together they
# cover all four coercions: the joint map, the shared vector coercion (directly
# and through the rgba path), ``apply_force``'s inline loop, and the shared
# gravity normalizer.
def _surfaces(sim):
    return {
        "set_joint_positions": lambda v: sim.set_joint_positions({"pan": v}, robot_name="arm"),
        "set_joint_velocities": lambda v: sim.set_joint_velocities({"pan": v}, robot_name="arm"),
        "apply_force.force": lambda v: sim.apply_force(body_name="link", force=[v, 0.0, 0.0]),
        "apply_force.torque": lambda v: sim.apply_force(body_name="link", torque=[v, 0.0, 0.0]),
        "apply_force.point": lambda v: sim.apply_force(body_name="link", force=[1.0, 0.0, 0.0], point=[v, 0.0, 0.0]),
        "set_geom_properties.color": lambda v: sim.set_geom_properties(geom_name="cube_geom", color=[v, 0.0, 0.0, 1.0]),
        "set_geom_properties.size": lambda v: sim.set_geom_properties(geom_name="cube_geom", size=[v, 0.1, 0.1]),
        "set_geom_properties.friction": lambda v: sim.set_geom_properties(
            geom_name="cube_geom", friction=[v, 0.005, 0.0001]
        ),
        "raycast.origin": lambda v: sim.raycast(origin=[v, 0.0, 1.0], direction=[0.0, 0.0, -1.0]),
        "multi_raycast.origin": lambda v: sim.multi_raycast(origin=[v, 0.0, 1.0], directions=[[0.0, 0.0, -1.0]]),
        "set_gravity.vector": lambda v: sim.set_gravity([v, 0.0, -9.81]),
        "set_gravity.scalar": lambda v: sim.set_gravity(v),
    }


class TestEveryRuntimeWriterRefusesABoolean:
    """No affected writer accepts a boolean, in any of its spellings."""

    @pytest.mark.parametrize("value", BOOLEANS)
    def test_a_boolean_is_refused_at_every_writer(self, sim, value):
        for label, call in _surfaces(sim).items():
            result = call(value)
            assert result["status"] == "error", f"{label} accepted {value!r}: {_text(result)}"
            # A vector surface names its elements ("must be numbers"), a scalar
            # one names the value ("must be a number") -- both must say which.
            assert re.search(r"must be (a )?numbers?", _text(result)), f"{label}: {_text(result)}"

    def test_the_refused_value_is_never_written(self, sim):
        """A refusal leaves the model exactly as it was -- no partial write."""
        sim.set_joint_positions({"pan": 0.4}, robot_name="arm")
        sim.set_gravity([0.0, 0.0, -9.81])
        model = sim._world._model
        before_qpos = float(sim.get_observation(robot_name="arm")["pan"])
        before_gravity = list(model.opt.gravity)
        before_rgba = list(model.geom_rgba[model.geom("cube_geom").id])

        assert sim.set_joint_positions({"pan": True}, robot_name="arm")["status"] == "error"
        assert sim.set_gravity([True, 0.0, -9.81])["status"] == "error"
        assert sim.set_gravity(True)["status"] == "error"
        assert sim.set_geom_properties(geom_name="cube_geom", color=[True, 0.0, 0.0, 1.0])["status"] == "error"

        assert float(sim.get_observation(robot_name="arm")["pan"]) == pytest.approx(before_qpos)
        assert list(model.opt.gravity) == before_gravity
        assert list(model.geom_rgba[model.geom("cube_geom").id]) == before_rgba

    def test_a_boolean_wrench_leaves_no_latched_force(self, sim):
        """``apply_force`` latches its wrench, so a refusal must latch nothing."""
        assert sim.apply_force(body_name="link", force=[True, 0.0, 0.0])["status"] == "error"
        assert not np.any(sim._world._data.xfrc_applied)


class TestTheAcceptedDomainIsUnchanged:
    """Every value these writers accepted before is still accepted."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(0.4, id="python-float"),
            pytest.param(np.float64(0.4), id="numpy-float64"),
            pytest.param(np.float32(0.25), id="numpy-float32"),
            pytest.param(1, id="python-int"),
            pytest.param(np.int64(1), id="numpy-int64"),
            pytest.param("0.5", id="numeric-string"),
        ],
    )
    def test_a_real_scalar_is_still_accepted(self, sim, value):
        for label, call in _surfaces(sim).items():
            if label == "set_gravity.scalar" and isinstance(value, str):
                # ``set_gravity`` accepts a scalar OR a 3-element vector, and a
                # 3-character string has ``len() == 3``, so it takes the vector
                # path and fails on the "." component. Pre-existing on main and
                # untouched here; the vector surfaces below still accept it.
                continue
            result = call(value)
            assert result["status"] == "success", f"{label} refused {value!r}: {_text(result)}"

    def test_numpy_vectors_are_still_accepted(self, sim):
        assert sim.apply_force(body_name="link", force=np.array([1.0, 0.0, 0.0]))["status"] == "success"
        assert (
            sim.set_geom_properties(geom_name="cube_geom", color=np.array([0.2, 0.4, 0.6, 1.0], dtype=np.float32))[
                "status"
            ]
            == "success"
        )
        assert (
            sim.raycast(origin=np.array([0.4, 0.0, 1.0]), direction=np.array([0.0, 0.0, -1.0]))["status"] == "success"
        )
        assert sim.set_gravity(np.array([0.0, 0.0, -9.81]))["status"] == "success"

    def test_an_explicit_zero_wrench_still_clears_a_latched_force(self, sim):
        """``[0, 0, 0]`` is the documented "stop this body" command, not a bool."""
        assert sim.apply_force(body_name="link", force=[2.0, 0.0, 0.0])["status"] == "success"
        assert np.any(sim._world._data.xfrc_applied)
        assert sim.apply_force(body_name="link", force=[0, 0, 0])["status"] == "success"
        assert not np.any(sim._world._data.xfrc_applied)


class TestTheEntryPointsAgree:
    """The direct API and the surfaces that already refused a boolean agree."""

    @pytest.mark.parametrize("value", BOOLEANS)
    def test_a_runtime_writer_agrees_with_the_scene_construction_vectors(self, sim, value):
        """``add_object``/``move_object`` already refused; the writers now match."""
        construction = sim.add_object(name=f"probe_{id(value)}", shape="box", position=[value, 0.0, 0.3])
        writer = sim.set_geom_properties(geom_name="cube_geom", color=[value, 0.0, 0.0, 1.0])
        assert construction["status"] == writer["status"] == "error"

    @pytest.mark.parametrize("value", BOOLEANS)
    def test_apply_force_agrees_with_the_agent_tool_router(self, sim, value):
        """The router refuses a bool component of this method's own vectors."""
        fn = getattr(type(sim), "apply_force")
        _, router_error = sim._validate_and_build_kwargs(
            "apply_force", "apply_force", inspect.signature(fn), {"body_name": "link", "force": [value, 0.0, 0.0]}
        )
        direct = sim.apply_force(body_name="link", force=[value, 0.0, 0.0])
        assert router_error is not None
        assert direct["status"] == "error"


class TestTheBooleanPredicateIsShared:
    """One predicate owns the rule, and every coercion applies it."""

    @pytest.mark.parametrize("value", BOOLEANS)
    def test_every_boolean_spelling_is_recognised(self, value):
        assert is_boolean(value) is True

    @pytest.mark.parametrize(
        "value", [0.0, 1.0, -2.5, 0, 1, np.float64(1.0), np.int64(0), np.float32(1.0), "x", None, np.array([1, 2])]
    )
    def test_a_non_boolean_is_not_recognised(self, value):
        assert is_boolean(value) is False

    def test_no_numeric_coercion_in_the_mujoco_physics_path_omits_the_predicate(self):
        """A coercion added later cannot silently reopen the hole.

        Each of these helpers turns caller-supplied values into the floats that
        reach a MuJoCo buffer. Without the shared predicate, ``float()`` accepts
        a boolean and the writer reports success for a value nobody asked for.
        """
        from strands_robots.simulation.mujoco import physics

        source = pathlib.Path(inspect.getfile(physics)).read_text()
        tree = ast.parse(source)
        coercions = {"_coerce_finite_vector", "_coerce_finite_joint_map"}
        seen = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in coercions:
                seen.add(node.name)
                calls = {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
                assert "is_boolean" in calls, f"{node.name} does not apply the shared boolean predicate"
        assert seen == coercions, f"expected coercions not found: {coercions - seen}"

    def test_the_apply_force_element_loop_applies_the_predicate(self):
        """``apply_force`` validates inline rather than through a helper."""
        from strands_robots.simulation.mujoco import physics

        body = inspect.getsource(physics.PhysicsMixin.apply_force)
        assert "is_boolean(_elem)" in body
