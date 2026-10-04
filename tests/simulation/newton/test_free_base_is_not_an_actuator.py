"""A floating base is not an actuator on the Newton backend.

Newton keeps one joint list per robot. A floating base's 6-DoF free joint is in
it and is not a commandable scalar - its coordinates are ``[xyz, quat_xyzw]``,
so there is no single target to write - which is why ``get_observation``,
``get_robot_state`` and the recording schema all skip it and surface the base as
the structured ``base_pos`` / ``base_quat`` / ``base_lin_vel`` / ``base_ang_vel``
signals instead. ``tests/simulation/newton/test_discovery_and_state.py`` pins
that exclusion for the state side; this module pins the action side of the same
contract, which ``robot_action_keys`` inherited from the base class unfiltered.

Four surfaces read that list, and every one of them was silent about the free
joint being in it:

* ``send_action`` accepted ``{free_joint: 0.5}`` under a ``success`` result and
  wrote a scalar target for a 6-DoF joint.
* the vector form of ``send_action`` refused an action of the width a recording
  actually holds, because the recorded columns exclude the free joint.
* ``SimEngine._coerce_action`` binds a vector positionally to this list.
* ``PolicyRunner.replay`` binds to it, so a floating-base episode aborted on
  frame 0 - the recording could be written but never replayed.

Solver-free: the engine is built via ``__new__`` with only the attributes these
paths read, so no Newton/Warp stack (or lerobot) is required and these pins run
on every CI job rather than being skipped.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_BASE_JOINT = "floating_base_joint"
_SCALAR_JOINTS = ["hip_yaw", "hip_pitch", "knee", "ankle"]


def _engine(*, free_base: bool) -> NewtonSimEngine:
    """A Newton engine bound to one robot, with or without a free root.

    ``__init__`` imports Newton/Warp and builds a solver; none of the action
    vocabulary or recording-schema paths touch physics, so the engine is
    constructed via ``__new__`` and given just the attributes they read - the
    same harness ``test_dataset_recording.py`` uses.
    """
    joints = ([_BASE_JOINT] if free_base else []) + _SCALAR_JOINTS
    world = SimWorld()
    world.robots["g1"] = SimRobot(name="g1", urdf_path="g1.xml", data_config="g1", joint_names=list(joints))

    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = object()  # non-None sentinel: "world created"
    engine.default_width = 64
    engine.default_height = 48
    engine._robot_free_base_joint = {"g1": _BASE_JOINT} if free_base else {}
    engine._lock = threading.RLock()
    engine._targets = {}
    engine._write_targets = lambda: None  # type: ignore[method-assign]
    engine._advance = lambda n_steps: None  # type: ignore[method-assign]
    return engine


class TestTheFreeBaseIsNotAnActionKey:
    """``robot_action_keys`` excludes the free root; joint names still carry it."""

    def test_the_free_base_joint_is_not_an_action_key(self):
        engine = _engine(free_base=True)
        assert engine.robot_action_keys("g1") == _SCALAR_JOINTS

    def test_the_joint_names_still_carry_it(self):
        """The state-side vocabulary is deliberately out of scope here.

        ``robot_joint_names`` has the same disagreement on the state side - it
        advertises a joint ``get_observation`` never emits - and it still names
        policy state keys, so narrowing it is a wider change than "the free
        joint is not an actuator" and is tracked separately. Asserted rather
        than left implicit so the two halves cannot be conflated by a later
        reader.

        The action-sizing consumer this docstring used to name is gone:
        ``training/rl``'s action head and checkpoint metadata read
        ``robot_action_keys``, so this list no longer sizes an action vector -
        see ``tests/training/test_rl_action_head_binds_action_keys.py``.
        """
        assert _engine(free_base=True).robot_joint_names("g1") == [_BASE_JOINT] + _SCALAR_JOINTS

    def test_a_fixed_base_robot_keeps_every_joint(self):
        """No free root, so the two vocabularies already agreed and still do."""
        engine = _engine(free_base=False)
        assert engine.robot_action_keys("g1") == _SCALAR_JOINTS
        assert engine.robot_action_keys("g1") == engine.robot_joint_names("g1")


class TestSendActionRefusesAScalarTargetOnTheFreeBase:
    """A 6-DoF joint cannot hold a scalar position target, so it is refused."""

    def test_the_free_base_key_is_refused_rather_than_written(self):
        engine = _engine(free_base=True)
        result = engine.send_action({_BASE_JOINT: 0.5}, robot_name="g1", n_substeps=1)

        assert result["status"] == "error", result
        # The value must not reach the target map: pre-fix this returned
        # "success" and wrote _targets[("g1", "floating_base_joint")] = 0.5.
        assert engine._targets == {}

    def test_the_refusal_names_the_key_and_the_commandable_set(self):
        engine = _engine(free_base=True)
        result = engine.send_action({_BASE_JOINT: 0.5}, robot_name="g1", n_substeps=1)

        payload = next(block["json"] for block in result["content"] if "json" in block)
        assert payload["unresolved_keys"] == [_BASE_JOINT]
        assert payload["applied"] == []
        text = result["content"][0]["text"]
        assert _BASE_JOINT in text
        # A free joint IS a joint, so the refusal says which category it fails.
        assert "not commandable joints" in text

    def test_the_scalar_joints_are_still_applied(self):
        engine = _engine(free_base=True)
        result = engine.send_action({"knee": 0.25}, robot_name="g1", n_substeps=1)

        assert result["status"] == "success", result
        assert engine._targets == {("g1", "knee"): 0.25}


class TestTheRecordedWidthRoundTrips:
    """The recorded action width and the action keys must be the same number.

    ``PolicyRunner.replay`` binds ``robot_action_keys`` against the recorded
    action vector and aborts the episode when the two widths differ, so this
    equality is the property that makes a floating-base recording replayable at
    all. Pinned as an equality between the two producers rather than assumed, so
    either side drifting fails here. The schema's action columns are now declared
    from ``robot_action_keys`` outright rather than inherited from the recorder's
    joint-name fallback, so these read the action slot of the schema tuple.
    """

    def test_the_declared_action_columns_equal_the_action_keys(self):
        engine = _engine(free_base=True)
        _joint_names, declared, *_rest = engine._collect_recording_schema()
        assert declared == engine.robot_action_keys("g1")

    def test_a_vector_action_of_the_recorded_width_is_accepted(self):
        """Pre-fix this refused: 4 recorded values against 5 advertised keys."""
        engine = _engine(free_base=True)
        _joint_names, declared, *_rest = engine._collect_recording_schema()
        recorded_frame = [0.1] * len(declared)

        result = engine.send_action(recorded_frame, robot_name="g1", n_substeps=1)

        assert result["status"] == "success", result
        assert engine._targets == {("g1", name): 0.1 for name in _SCALAR_JOINTS}


class _TargetBuffer:
    """The slice of a Warp array ``_write_targets`` reads back."""

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def numpy(self) -> list[float]:
        return list(self.values)


@pytest.mark.parametrize(
    ("coord_layout", "width", "knee_index"),
    [
        # Newton's default: one target per DOF; the free base spans 6, so the
        # first scalar joint (hip_yaw) is DOF 6 and the knee DOF 8.
        (False, 6 + len(_SCALAR_JOINTS), 8),
        # ``newton.use_coord_layout_targets``: one target per coordinate; the
        # free base spans 7 (xyz + quaternion), so the knee is coordinate 9.
        (True, 7 + len(_SCALAR_JOINTS), 9),
    ],
)
def test_a_target_lands_on_its_own_joint_behind_a_free_base(coord_layout, width, knee_index):
    """Pre-fix every target was written at the coordinate index, so on the
    default DOF layout the knee's target drove the ankle and the last joint's
    target fell off the end of the array - a walking policy toppled whatever
    it commanded."""
    engine = _engine(free_base=True)
    engine._joint_coord_index = {("g1", name): 7 + i for i, name in enumerate(_SCALAR_JOINTS)}
    engine._joint_dof_index = {("g1", name): 6 + i for i, name in enumerate(_SCALAR_JOINTS)}
    engine._model = SimpleNamespace(use_coord_layout_targets=coord_layout, device="cpu")
    engine._control = SimpleNamespace(joint_target_q=_TargetBuffer([0.0] * width))
    engine._wp = SimpleNamespace(float32="float32", array=lambda values, dtype, device: _TargetBuffer(values))
    engine._targets = {("g1", name): float(i + 1) for i, name in enumerate(_SCALAR_JOINTS)}

    NewtonSimEngine._write_targets(engine)

    written = engine._control.joint_target_q.values
    first = knee_index - _SCALAR_JOINTS.index("knee")
    assert written[first:] == [1.0, 2.0, 3.0, 4.0]
    assert written[:first] == [0.0] * first
