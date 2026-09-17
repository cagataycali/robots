"""A bare entity name several robots carry is refused by every physics write.

``_resolve_mj_name`` tries a caller's name verbatim and then under each robot's
namespace, returning the FIRST match. Its docstring calls that "a deliberate
'unambiguous or explicit' contract" and says the caller MUST qualify the name in
a multi-robot scene - but only the readers it was written for could live with
it. Five ``PhysicsMixin`` writes reach the same retry, and in a scene with two
so101s each of them wrote the first robot attached and reported success, while
the list form, ``get_robot_state``, ``move_to`` and ``run_policy`` on that same
scene all refused to guess (#2453 / #2549 closed those doors, not these):

==============================  =========================  ==================
write, no robot_name            before                     after
==============================  =========================  ==================
``set_joint_positions({"1":})`` moved ``so101`` silently    refused
``set_joint_velocities``        moved ``so101`` silently    refused
``apply_force("gripper")``      latched on ``so101``        refused
``set_body_properties``         re-massed ``so101``         refused
``set_geom_properties``         recoloured the first robot  refused
``get_body_state`` and the      answered about the first    answers, and names
other two readers               robot, naming nobody        the entity it read
==============================  =========================  ==================

The readers stay on the fallback deliberately: a read changes nothing and can be
asked again, and they are the population the retry was written for. That premise
is what the second half of this file pins - because the answer named the
*request* (``Body 'base' (id=1)``, in a scene whose only bodies are
``alice/base`` and ``bob/base``), so it named neither the entity that had been
read nor the fact that a choice had been made, and "ask again" was not a remedy
a caller could see. Each read now appends the name it resolved to, plus - when
several robots carry it - who else carries it and the spelling that reads them.

The refusal fires only when the retry is what decides. A name the model carries
verbatim, and robots that resolve a name to the SAME joint, are not ambiguous -
and a namespace-less robot offers no qualified spelling, so it cannot make a
unique name unwritable. GL-free (``mesh=False``, no rendering).
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco as mj  # noqa: E402

from strands_robots.simulation.mujoco.physics import PhysicsMixin  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine, Simulation  # noqa: E402

_ARM = """<mujoco model="arm"><compiler angle="radian"/>
 <worldbody><body name="base" pos="0 0 0.05"><geom name="pad" type="box" size="0.05 0.05 0.05"/>
  <body name="link" pos="0 0 0.06"><joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="4"/>
   <geom name="rod" type="capsule" fromto="0 0 0 0.18 0 0" size="0.02"/>{extra}</body></body></worldbody>
 <actuator><position name="a_pan" joint="pan" kp="50" ctrlrange="-2 2"/></actuator></mujoco>
"""
# Only the second robot carries ``tip`` / ``spin`` / ``tip_pad``: the case the
# namespace retry exists for, which must keep resolving.
_TIP = """<body name="tip" pos="0.18 0 0"><joint name="spin" type="hinge" axis="1 0 0" range="-2 2"/>
   <geom name="tip_pad" type="sphere" size="0.03"/></body>"""


@dataclass(frozen=True)
class _Door:
    """One physics write, plus the names it needs to be driven."""

    shared: str  # an entity name BOTH robots carry
    unique: str  # an entity name only ``bob`` carries
    kind: str  # how the refusal spells what the name names
    obj: int  # the mjtObj the name names, to check the spellings offered exist
    call: Callable[[Any, str], dict[str, Any]]


_JOINT, _BODY, _GEOM = mj.mjtObj.mjOBJ_JOINT, mj.mjtObj.mjOBJ_BODY, mj.mjtObj.mjOBJ_GEOM

WRITES: dict[str, _Door] = {
    "set_joint_positions": _Door("pan", "spin", "joint key", _JOINT, lambda s, n: s.set_joint_positions({n: 0.3})),
    "set_joint_velocities": _Door("pan", "spin", "joint key", _JOINT, lambda s, n: s.set_joint_velocities({n: 0.5})),
    "apply_force": _Door("base", "tip", "body", _BODY, lambda s, n: s.apply_force(n, force=[0.0, 0.0, 1.0])),
    "set_body_properties": _Door("base", "tip", "body", _BODY, lambda s, n: s.set_body_properties(n, mass=0.5)),
    "set_geom_properties": _Door(
        "pad", "tip_pad", "geom", _GEOM, lambda s, n: s.set_geom_properties(n, color=[1, 0, 0, 1])
    ),
}
DOORS = sorted(WRITES)


@dataclass(frozen=True)
class _Read:
    """One physics read, plus the names it needs to be driven."""

    shared: str  # an entity name BOTH robots carry
    unique: str  # an entity name only ``bob`` carries
    obj: int  # the mjtObj the name names, to check the spellings offered exist
    verbatim: str  # what ``add_object(name="widget")`` names of this type
    call: Callable[[Any, str], dict[str, Any]]


# The geom row is why each ``get_jacobian`` branch passes its own ``mjtObj``: a
# geom id read back as a body names a different entity, or none at all.
READS: dict[str, _Read] = {
    "get_body_state": _Read("base", "tip", _BODY, "widget", lambda s, n: s.get_body_state(n)),
    "get_jacobian": _Read("base", "tip", _BODY, "widget", lambda s, n: s.get_jacobian(body_name=n)),
    "get_jacobian_geom": _Read("pad", "tip_pad", _GEOM, "widget_geom", lambda s, n: s.get_jacobian(geom_name=n)),
    "forward_kinematics": _Read("base", "tip", _BODY, "widget", lambda s, n: s.forward_kinematics(n)),
}
WINDOWS = sorted(READS)


def _scene(tmp_path, *robots: str) -> Simulation:
    """A world with one robot per name; the second carries the extra ``tip`` chain."""
    sim = Simulation(tool_name="test_ambiguous_bare_name", mesh=False)
    assert sim.create_world(gravity=[0, 0, 0])["status"] == "success"
    for index, name in enumerate(robots):
        path = tmp_path / f"{name}.xml"
        path.write_text(_ARM.format(extra=_TIP if index else ""), encoding="utf-8")
        result = sim.add_robot(name=name, urdf_path=str(path), position=[0.5 * index, 0, 0])
        assert result["status"] == "success", result
    return sim


@pytest.fixture
def two_arms(tmp_path):
    sim = _scene(tmp_path, "alice", "bob")
    yield sim
    sim.cleanup()


@pytest.fixture
def one_arm(tmp_path):
    sim = _scene(tmp_path, "alice")
    yield sim
    sim.cleanup()


def _text(result: dict[str, Any]) -> str:
    return "\n".join(block["text"] for block in result["content"] if "text" in block)


def _state(sim: Simulation) -> list[Any]:
    """Everything the five writes can touch, so "nothing was written" is measured."""
    world = sim._world
    assert world is not None, "the fixture built a world"
    model, data = world._model, world._data
    return [
        data.qpos.copy(),
        data.qvel.copy(),
        data.ctrl.copy(),
        data.xfrc_applied.copy(),
        model.body_mass.copy(),
        model.body_inertia.copy(),
        model.geom_rgba.copy(),
        model.geom_friction.copy(),
        model.geom_size.copy(),
    ]


def _unchanged(before: list[Any], after: list[Any]) -> bool:
    return all((b == a).all() for b, a in zip(before, after, strict=True))


@pytest.mark.parametrize("door", DOORS)
def test_a_bare_name_both_robots_carry_is_refused_and_writes_nothing(two_arms, door):
    """The defect: each write moved the first robot attached and said success."""
    before = _state(two_arms)
    result = WRITES[door].call(two_arms, WRITES[door].shared)
    assert result["status"] == "error", result
    assert _unchanged(before, _state(two_arms)), f"{door} wrote state it refused"


@pytest.mark.parametrize("door", DOORS)
def test_the_refusal_names_both_owners_and_only_spellings_that_exist(two_arms, door):
    """Both owners, in attachment order, and every offered spelling resolves.

    Naming one owner reads as a recommendation, and an agent then picks the very
    robot the fallback would have. Offering a spelling the model does not carry
    is worse: the caller's only escape route fails too.
    """
    spec = WRITES[door]
    text = _text(spec.call(two_arms, spec.shared))
    assert text.startswith(
        f"{door}: {spec.kind} '{spec.shared}' is ambiguous - robots 'alice' and 'bob' each carry it, "
        f"so nothing was written."
    ), text
    for owner in ("alice", "bob"):
        qualified = f"{owner}/{spec.shared}"
        assert f"'{qualified}'" in text, text
        assert mj.mj_name2id(two_arms._world._model, spec.obj, qualified) >= 0, qualified


@pytest.mark.parametrize("door", DOORS)
def test_a_qualified_name_still_writes(two_arms, door):
    """Control: the remedy the refusal offers is the one that works."""
    spec = WRITES[door]
    before = _state(two_arms)
    assert spec.call(two_arms, f"bob/{spec.shared}")["status"] == "success"
    assert not _unchanged(before, _state(two_arms))


@pytest.mark.parametrize("door", DOORS)
def test_a_bare_name_only_one_robot_carries_still_writes(two_arms, door):
    """Control: the retry keeps serving the case it was written for."""
    spec = WRITES[door]
    assert spec.call(two_arms, spec.unique)["status"] == "success"


@pytest.mark.parametrize("door", DOORS)
def test_a_single_robot_scene_is_unchanged(one_arm, door):
    """Control: one robot means one answer, so nothing is refused."""
    spec = WRITES[door]
    assert spec.call(one_arm, spec.shared)["status"] == "success"


@pytest.mark.parametrize("door", DOORS)
def test_an_unknown_name_keeps_its_own_refusal(two_arms, door):
    """Control: the guard refuses a conflict, not any name a caller invents."""
    result = WRITES[door].call(two_arms, "no_such_entity")
    assert result["status"] == "error", result
    assert "is ambiguous" not in _text(result), _text(result)


def test_a_name_the_scene_carries_verbatim_is_still_written(two_arms):
    """An object named like a robot body is written, not refused.

    ``_resolve_mj_name`` answers a name the model carries verbatim before it
    tries any namespace, so an ``add_object(name="base")`` beside two arms that
    each have a ``base`` is not ambiguous at all: one entity is named, and it is
    the one the write lands on. Counting the two arms as owners here would
    refuse a call that has exactly one correct answer.
    """
    assert two_arms.add_object(name="base", shape="box", size=[0.05] * 3, position=[0, 0.6, 0.1])["status"] == "success"
    model = two_arms._world._model
    assert mj.mj_name2id(model, _BODY, "base") >= 0  # premise: the bare name is a body of its own
    result = two_arms.apply_force("base", force=[0.0, 0.0, 1.0])
    assert result["status"] == "success", result
    touched = [
        mj.mj_id2name(model, _BODY, int(bid))
        for bid in np.nonzero(np.abs(two_arms._world._data.xfrc_applied).sum(axis=1))[0]
    ]
    assert touched == ["base"], touched


@pytest.mark.parametrize("window", WINDOWS)
def test_a_bare_name_both_robots_carry_is_still_read(two_arms, window):
    """Deliberate: the readers are the population the first-match retry serves.

    They change nothing and can be asked again with a qualified name, so they
    keep the pre-namespacing UX ``test_body_readers_resolve_one_name`` pins.
    """
    spec = READS[window]
    assert spec.call(two_arms, spec.shared)["status"] == "success"


@pytest.mark.parametrize("window", WINDOWS)
def test_a_resolved_read_names_the_entity_it_answered_about(two_arms, window):
    """The defect: the answer echoed the caller's bare name and named nobody."""
    spec = READS[window]
    text = _text(spec.call(two_arms, spec.shared))
    assert f"resolved '{spec.shared}' to 'alice/{spec.shared}'" in text, text


@pytest.mark.parametrize("window", WINDOWS)
def test_the_note_names_every_owner_and_only_spellings_that_exist(two_arms, window):
    """Both owners, and the spelling offered is the one the caller does not have.

    Offering the entity that just answered is not a remedy, and offering a name
    the model does not carry sends the caller down a route that fails too.
    """
    spec = READS[window]
    text = _text(spec.call(two_arms, spec.shared))
    assert f"robots 'alice' and 'bob' each carry '{spec.shared}'" in text, text
    # The remedy list itself, not the sentence around it: an offer that includes
    # the entity the caller was just handed is not a route to the other one.
    offered = text.rsplit("qualify the name to read another ", 1)[1]
    assert f"'bob/{spec.shared}'" in offered, offered
    assert f"'alice/{spec.shared}'" not in offered, offered
    assert mj.mj_name2id(two_arms._world._model, spec.obj, f"bob/{spec.shared}") >= 0


@pytest.mark.parametrize("window", WINDOWS)
def test_the_spelling_the_note_offers_reads_the_other_robot(two_arms, window):
    """Control: the remedy works, and an answer the caller named needs no note."""
    spec = READS[window]
    result = spec.call(two_arms, f"bob/{spec.shared}")
    assert result["status"] == "success", result
    assert "resolved" not in _text(result), _text(result)


def test_the_named_entity_is_the_one_whose_state_was_returned(two_arms):
    """The note and the numbers agree: ``alice/base``, not ``bob/base``.

    The resolved name is read back off the id the answer describes, so a note
    naming the robot the answer is *not* about - a worse dead end than naming
    nobody - is not reachable by the note and the read disagreeing. The two arms
    stand half a metre apart, so the positions tell them apart.
    """
    bare = two_arms.get_body_state("base")
    assert "resolved 'base' to 'alice/base'" in _text(bare)
    position = next(block["json"]["position"] for block in bare["content"] if "json" in block)
    named = two_arms.get_body_state("alice/base")["content"][1]["json"]["position"]
    rival = two_arms.get_body_state("bob/base")["content"][1]["json"]["position"]
    assert position == pytest.approx(named)
    assert position != pytest.approx(rival)


@pytest.mark.parametrize("window", WINDOWS)
def test_a_bare_name_one_robot_carries_is_named_without_rivals(two_arms, window):
    """Control: the retry still serves the unambiguous case, with no rival list."""
    spec = READS[window]
    text = _text(spec.call(two_arms, spec.unique))
    assert f"resolved '{spec.unique}' to 'bob/{spec.unique}'." in text, text
    assert "each carry" not in text, text


@pytest.mark.parametrize("window", WINDOWS)
def test_a_name_the_scene_carries_verbatim_is_read_without_a_note(two_arms, window):
    """Control: the note reports the retry, and the retry did not run here.

    An ``add_object`` is answered by the verbatim lookup before any namespace is
    tried, so the caller's name IS the entity's name and there is nothing to
    disambiguate. The name is per type rather than per object, because
    ``add_object(name="widget")`` builds the body ``widget`` and the geom
    ``widget_geom`` - so the premise is asserted rather than assumed.
    """
    spec = READS[window]
    added = two_arms.add_object(name="widget", shape="box", size=[0.05] * 3, position=[0, 0.6, 0.1])
    assert added["status"] == "success", added
    assert mj.mj_name2id(two_arms._world._model, spec.obj, spec.verbatim) >= 0, spec.verbatim
    result = spec.call(two_arms, spec.verbatim)
    assert result["status"] == "success", result
    assert "resolved" not in _text(result), _text(result)


def test_a_joint_name_one_robot_carries_outranks_another_robot_label(tmp_path):
    """A namespace hit ends the lookup, so a label on a second robot is not a rival.

    The write path resolves ``<namespace><key>`` for every robot first and only
    falls back to registry labels when none carried the key. Consulting labels
    anyway would call this scene ambiguous - the so101's joint ``1`` is labelled
    ``shoulder_pan``, and the other arm has a joint of that literal name - while
    resolution has exactly one answer, and it is the joint.
    """
    arm = tmp_path / "labelled.xml"
    arm.write_text(_ARM.replace('name="pan"', 'name="shoulder_pan"'), encoding="utf-8")
    sim = MuJoCoSimEngine(tool_name="test_ambiguous_bare_name_precedence", mesh=False)
    sim.create_world()
    try:
        assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
        assert sim.add_robot(name="arm2", urdf_path=str(arm), position=[0.5, 0, 0])["status"] == "success"
        model, data = sim._world._model, sim._world._data
        assert sim.set_joint_positions({"shoulder_pan": 0.3})["status"] == "success"
        moved = data.qpos[model.jnt_qposadr[mj.mj_name2id(model, _JOINT, "arm2/shoulder_pan")]]
        held = data.qpos[model.jnt_qposadr[mj.mj_name2id(model, _JOINT, "so101/1")]]
        assert (moved, held) == pytest.approx((0.3, 0.0))
    finally:
        sim.cleanup()


def test_namespace_less_robots_do_not_make_a_unique_name_unwritable(tmp_path):
    """A key naming exactly one joint in the model is written, not refused.

    Two robots registered without a namespace - what the Newton backend builds,
    and what ``TestDirectJointControlListForm`` registers - both "carry" every
    bare joint name, because ``<namespace><key>`` is the key itself for each of
    them. Counting them as owners refused ``{"shoulder": 0.1}`` in a scene whose
    model has exactly ONE joint named ``shoulder``, and pointed the caller at
    ``'arm_a/shoulder'``, which names nothing at all - a refusal whose own
    remedy cannot be followed. The retry skips a robot with no namespace, so the
    guard that mirrors it skips it too.
    """
    from strands_robots.simulation.models import SimRobot, SimStatus, SimWorld

    sim = Simulation(tool_name="test_ambiguous_bare_name_ns_less", mesh=False)
    sim._world = SimWorld()
    spec = mj.MjSpec.from_string(_ARM.format(extra=""))
    sim._world._backend_state["spec"] = spec
    sim._world._model = spec.compile()
    sim._world._data = mj.MjData(sim._world._model)
    sim._world.status = SimStatus.IDLE
    try:
        for name in ("arm_a", "arm_b"):
            sim._world.robots[name] = SimRobot(name=name, urdf_path="", joint_names=["pan"], namespace="")
        assert sim._world._model.njnt == 1  # premise: one joint named 'pan' in the whole model
        assert sim.set_joint_positions({"pan": 0.1})["status"] == "success"
    finally:
        sim.cleanup()


@pytest.mark.parametrize("key", ["1", "shoulder_pan", "Shoulder_Pan"])
def test_the_reported_so101_scene_refuses_by_joint_name_and_by_registry_label(key):
    """The scene from the report: two so101s, keyed by servo id and by label.

    A registry joint label (``shoulder_pan`` for the SO-101's joint ``1``, in any
    case) is the other spelling of the same write, and both so101s carry it.
    """
    sim = MuJoCoSimEngine(tool_name="test_ambiguous_bare_name_so101", mesh=False)
    sim.create_world()
    try:
        assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
        assert sim.add_robot(name="arm2", data_config="so101", position=[0.5, 0, 0])["status"] == "success"
        result = sim.set_joint_positions({key: 0.3})
        assert result["status"] == "error", result
        assert _text(result).startswith(
            f"set_joint_positions: joint key '{key}' is ambiguous - robots 'so101' and 'arm2' each carry it"
        ), _text(result)
        assert f"qualify the key ('so101/{key}' or 'arm2/{key}')." in _text(result)
        assert sim.set_joint_positions({key: 0.3}, robot_name="arm2")["status"] == "success"
    finally:
        sim.cleanup()


def _mixin_methods() -> list[ast.FunctionDef]:
    """Every method defined on ``PhysicsMixin``, parsed from its own source."""
    source = inspect.getsourcefile(PhysicsMixin)
    assert source is not None
    tree = ast.parse(pathlib.Path(source).read_text(encoding="utf-8"))
    mixin = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == PhysicsMixin.__name__)
    return [node for node in mixin.body if isinstance(node, ast.FunctionDef)]


def _calls(method: ast.FunctionDef, name: str) -> bool:
    """Whether ``method`` calls ``self.<name>`` anywhere in its body."""
    return any(
        isinstance(node.func, ast.Attribute) and node.func.attr == name
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
    )


def _writes_state(method: ast.FunctionDef) -> bool:
    """Whether ``method`` assigns into the compiled model or its data."""
    for node in ast.walk(method):
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AugAssign):
            targets = [node.target]
        for target in targets:
            if ast.unparse(target).startswith(("model.", "data.", "self._world._model.", "self._world._data.")):
                return True
    return False


def _writes_resolving_a_name() -> dict[str, bool]:
    """Every ``PhysicsMixin`` method that resolves a caller's name AND writes state.

    Derived from the source, so a sixth write inherits the rule instead of
    quietly re-opening the door: a method is in scope when it resolves a name
    (``_resolve_mj_name``, or ``_resolve_joint_write_targets`` which resolves
    the dict-form joint keys) and assigns into ``model``/``data``, and it
    complies when the ambiguity guard is reached - directly, or through the
    joint-key resolver that carries it.
    """
    found: dict[str, bool] = {}
    for method in _mixin_methods():
        via_joint_keys = _calls(method, "_resolve_joint_write_targets")
        if not (_calls(method, "_resolve_mj_name") or via_joint_keys) or not _writes_state(method):
            continue
        found[method.name] = via_joint_keys or _calls(method, "_refuse_ambiguous_bare_name")
    return found


def test_every_physics_write_that_resolves_a_name_enforces_the_contract():
    """The family itself, so the next write does not re-open the door."""
    writes = _writes_resolving_a_name()
    assert set(writes) == set(WRITES), "the table above no longer covers the mixin's writes"
    assert [name for name, guarded in writes.items() if not guarded] == []


def _reads_resolving_a_name() -> dict[str, bool]:
    """Every caller-facing ``PhysicsMixin`` method that resolves a name and reads.

    Derived from the source so a fourth reader inherits the rule instead of
    silently going back to naming the request: a method is in scope when it
    resolves a name through ``_resolve_mj_name`` and assigns into neither the
    model nor its data, and it complies when it reaches ``_resolved_name_note``.

    Scoped to public methods, because the private ones the filter also catches
    (``_resolve_joint_label``, ``_robot_joint_labels``,
    ``_resolve_joint_write_targets``) resolve on behalf of a verb that answers -
    and that verb carries the note, or the refusal, itself.
    """
    return {
        method.name: _calls(method, "_resolved_name_note")
        for method in _mixin_methods()
        if not method.name.startswith("_") and _calls(method, "_resolve_mj_name") and not _writes_state(method)
    }


def test_every_physics_read_that_resolves_a_name_names_the_entity():
    """The read family itself, so the next reader does not name the request."""
    reads = _reads_resolving_a_name()
    assert set(reads) == {"get_body_state", "get_jacobian", "forward_kinematics"}, (
        "the read table above no longer covers the mixin's resolving readers"
    )
    assert [name for name, named in reads.items() if not named] == []
