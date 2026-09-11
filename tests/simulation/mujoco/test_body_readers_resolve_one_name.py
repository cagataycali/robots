"""One body name is enough: every body reader resolves it the same way.

``add_robot`` compiles a robot under its own namespace, so the bodies in the
scene are ``arm/base``, ``arm/link`` - never the bare ``base`` a caller reads
off an MJCF, a joint name or a ``list_bodies`` hint for a single-robot scene.
:meth:`~strands_robots.simulation.mujoco.physics.PhysicsMixin._resolve_mj_name`
exists for exactly that gap: it tries the name verbatim, then retries it under
every robot's namespace, and its docstring names the population it serves -
"physics/introspection methods that accept raw body/joint/site names
(``get_body_state("gripper")`` etc.)".

Five methods of that mixin take a caller-supplied ``body_name`` and look up
``mjOBJ_BODY`` with it. Four routed it through the shared resolver;
``forward_kinematics`` called ``mj_name_to_id`` directly, so the one name that
worked in ``get_body_state`` / ``get_jacobian`` / ``apply_force`` /
``set_body_properties`` was refused by the fifth:

=======================  ==================  ==================
body_name                four readers        forward_kinematics
=======================  ==================  ==================
``"base"`` (bare)        success             **Body not found**
``"arm/base"``           success             success
``"nope"``               not found           not found
=======================  ==================  ==================

The table below is the whole relation, driven over the derived family rather
than a hand-written list, so the bare-name row is the regression and the other
two rows are the controls: the namespaced name must keep working (nothing was
traded away) and an unknown name must still be refused by all five (the retry
resolves a namespace, not any name at all).

The last test pins the family itself. A sixth body reader added later inherits
the rule instead of quietly re-introducing the split: any ``PhysicsMixin``
method that takes ``body_name`` and looks up ``mjOBJ_BODY`` must reach the
lookup through ``_resolve_mj_name``. GL-free (``mesh=False``, no rendering).
"""

import ast
import inspect
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.physics import PhysicsMixin  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

ARM_MJCF = """<mujoco model="arm"><compiler angle="radian"/>
 <worldbody><body name="base" pos="0 0 0.05"><geom type="box" size="0.05 0.05 0.05"/>
  <body name="link" pos="0 0 0.06"><joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="4"/>
   <geom type="capsule" fromto="0 0 0 0.18 0 0" size="0.02"/></body></body></worldbody>
 <actuator><position name="a_pan" joint="pan" kp="50" ctrlrange="-2 2"/></actuator></mujoco>
"""

# Every PhysicsMixin method that reads a caller-supplied body name. Each call
# is the cheapest one that reaches the lookup and nothing else.
BODY_READERS: dict[str, Callable[[Simulation, Any], dict[str, Any]]] = {
    "get_body_state": lambda sim, name: sim.get_body_state(name),
    "get_jacobian": lambda sim, name: sim.get_jacobian(body_name=name),
    "apply_force": lambda sim, name: sim.apply_force(name, force=[0.0, 0.0, 0.0]),
    "set_body_properties": lambda sim, name: sim.set_body_properties(name, mass=0.25),
    "forward_kinematics": lambda sim, name: sim.forward_kinematics(name),
}


@pytest.fixture
def sim(tmp_path):
    """A single-robot scene whose bodies are all namespaced ``arm/...``."""
    arm = tmp_path / "arm.xml"
    arm.write_text(ARM_MJCF, encoding="utf-8")
    simulation = Simulation(tool_name="test_body_readers_resolve_one_name", mesh=False)
    assert simulation.create_world(gravity=[0, 0, 0])["status"] == "success"
    assert simulation.add_robot(name="arm", urdf_path=str(arm))["status"] == "success"
    yield simulation
    simulation.cleanup()


def _bodies(sim: Simulation) -> list[str]:
    result = sim.list_bodies()
    assert result["status"] == "success", result
    return list(next(block["json"]["bodies"] for block in result["content"] if "json" in block))


def test_the_scene_namespaces_its_bodies(sim):
    """Guard the premise: without a namespace there is no bare name to resolve."""
    bodies = _bodies(sim)
    assert "arm/base" in bodies, bodies
    assert "base" not in bodies, bodies


@pytest.mark.parametrize("reader", sorted(BODY_READERS))
def test_a_bare_body_name_is_resolved_by_every_reader(sim, reader):
    """The regression: the namespace retry serves the whole family, not four fifths."""
    assert BODY_READERS[reader](sim, "base")["status"] == "success"


@pytest.mark.parametrize("reader", sorted(BODY_READERS))
def test_a_namespaced_body_name_is_still_read_by_every_reader(sim, reader):
    """Control: resolving a bare name costs the explicit name nothing."""
    assert BODY_READERS[reader](sim, "arm/base")["status"] == "success"


@pytest.mark.parametrize("reader", sorted(BODY_READERS))
def test_an_unknown_body_name_is_still_refused_by_every_reader(sim, reader):
    """Control: the retry resolves a namespace, not any name a caller invents."""
    result = BODY_READERS[reader](sim, "no_such_body")
    assert result["status"] == "error", result
    text = next(block["text"] for block in result["content"] if "text" in block)
    assert "no_such_body" in text and "not found" in text


def test_forward_kinematics_reports_the_body_it_was_asked_for(sim):
    """The resolved read answers about the requested name, as the siblings do."""
    result = sim.forward_kinematics("base")
    payload = next(block["json"] for block in result["content"] if "json" in block)
    assert payload["body"] == "base"
    assert payload["position"] == pytest.approx(sim.get_body_state("arm/base")["content"][1]["json"]["position"])


def _body_readers_in_source() -> dict[str, bool]:
    """Every ``PhysicsMixin`` method taking ``body_name`` -> does it use the resolver.

    Derived from the source so the family cannot drift from the table above: a
    method is in scope when it declares a ``body_name`` parameter and mentions
    ``mjOBJ_BODY``, and it complies when that body lookup goes through
    ``_resolve_mj_name`` rather than ``mj_name_to_id``.
    """
    source_file = inspect.getsourcefile(PhysicsMixin)
    assert source_file is not None
    tree = ast.parse(pathlib.Path(source_file).read_text(encoding="utf-8"))
    mixin = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == PhysicsMixin.__name__
    )
    found: dict[str, bool] = {}
    for method in [node for node in mixin.body if isinstance(node, ast.FunctionDef)]:
        params = {arg.arg for arg in method.args.args + method.args.kwonlyargs}
        source = ast.unparse(method)
        if "body_name" not in params or "mjOBJ_BODY" not in source:
            continue
        found[method.name] = any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "_resolve_mj_name"
            and any("mjOBJ_BODY" in ast.unparse(arg) for arg in call.args)
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
        )
    return found


def test_every_body_reader_resolves_through_the_shared_lookup():
    """A sixth reader inherits the rule instead of re-opening the split."""
    readers = _body_readers_in_source()
    assert set(readers) == set(BODY_READERS), "the table above no longer covers the mixin's body readers"
    assert [name for name, ok in readers.items() if not ok] == []
