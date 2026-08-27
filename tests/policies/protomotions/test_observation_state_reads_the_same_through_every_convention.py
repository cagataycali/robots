"""Every ``observation.state`` spelling resolves to the same G1 pose.

:meth:`ProtoMotionsPolicy._pack_by_name` offers three observation conventions
and two of them read the same key, ``observation.state``: convention 1 pairs it
with a ``state_keys`` list supplied on the observation, convention 3 pairs it
with the policy's own ``_robot_state_keys``. Both index the array positionally,
so both need it flat -- and a runtime that batches its state feeds ``(1, D)``
rather than ``(D,)``, which is exactly what LeRobot's
``AddBatchDimensionObservationStep`` produces.

Flattening in only one of the two branches made the pair disagree about one
observation: adding the documented ``state_keys`` list turned a readable
batched state into ``TypeError: only length-1 arrays can be converted to
Python scalars`` -- a message naming neither the key, the shape nor a remedy,
and a type outside :meth:`ProtoMotionsPolicy.get_actions`'s declared
``Raises``. The grid below drives the public ``get_actions`` over four
spellings of one state vector crossed with both conventions and demands a
single answer, so the normalization cannot drift back into one branch.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.protomotions import (
    GTP_G1_JOINT_NAMES,
    ProtoMotionsPolicy,
)

JOINT_NAMES: tuple[str, ...] = GTP_G1_JOINT_NAMES
NUM_DOFS = len(JOINT_NAMES)

# A pose whose every entry is distinct, so a mis-ordered or partially-filled
# read cannot coincide with the right answer.
POSE = np.linspace(-0.4, 0.4, NUM_DOFS).astype(np.float32)


class _EchoSession:
    """A tracker stub that returns the resolved joint positions as its targets.

    Echoing makes the action dict the policy returns *be* the state the policy
    read out of the observation, so a test can compare observations by
    comparing outputs without owning any ONNX weights.
    """

    def run(self, output_names: list[str] | None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        pos = inputs["current_dof_pos"]
        answers = {
            "actions": pos.copy(),
            "joint_pos_targets": pos.copy(),
            "stiffness_targets": np.full_like(pos, 40.0),
            "damping_targets": np.full_like(pos, 2.5),
        }
        return [answers[name] for name in (output_names or list(answers))]


def _flat_cache(num_frames: int = 20) -> dict[str, Any]:
    """A zero reference motion: the tracker's future window is not under test."""
    num_bodies = 33
    return {
        "dof_pos": np.zeros((num_frames, NUM_DOFS), dtype=np.float32),
        "dof_vel": np.zeros((num_frames, NUM_DOFS), dtype=np.float32),
        "body_rot": np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (num_frames, num_bodies, 1)),
        "body_pos": np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        "body_vel": np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        "body_ang_vel": np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        "control_dt": 0.02,
        "num_frames": num_frames,
    }


def _policy() -> ProtoMotionsPolicy:
    return ProtoMotionsPolicy(session=_EchoSession(), motion=_flat_cache())


# The frame inputs ``observation.state`` cannot carry. Supplying them as kwargs
# keeps every cell of the grid about the joint-position read alone.
_FRAME_KWARGS: dict[str, Any] = {
    "anchor_rot_xyzw": [0.0, 0.0, 0.0, 1.0],
    "root_ang_vel_local": [0.0, 0.0, 0.0],
    "dof_vel": np.zeros(NUM_DOFS, dtype=np.float32),
}


def _resolved_pose(observation: dict[str, Any], policy: ProtoMotionsPolicy | None = None) -> np.ndarray:
    """Return the joint-target vector ``get_actions`` derives from ``observation``."""
    pol = policy if policy is not None else _policy()
    action = asyncio.run(pol.get_actions(observation, "", **_FRAME_KWARGS))[0]
    return np.array([action[name] for name in JOINT_NAMES], dtype=np.float32)


# Four spellings of one state vector. ``reshape(-1)`` is the only thing that
# makes them interchangeable, which is what the grid measures.
_SPELLINGS: dict[str, Any] = {
    "flat_array": POSE,
    "batched_row": POSE.reshape(1, NUM_DOFS),
    "column": POSE.reshape(NUM_DOFS, 1),
    "nested_list": [POSE.tolist()],
}


class TestPremisesOfTheSpellingGrid:
    """The grid is only meaningful if its spellings really differ in shape."""

    def test_every_spelling_carries_the_same_values_in_a_different_shape(self):
        shapes = set()
        for name, spelling in _SPELLINGS.items():
            arr = np.asarray(spelling, dtype=np.float32)
            shapes.add(arr.shape)
            assert np.array_equal(arr.reshape(-1), POSE), f"{name} does not carry POSE"
        # Three distinct shapes across four spellings: the two list-shaped and
        # array-shaped batched rows coincide, everything else differs.
        assert len(shapes) == 3, shapes
        assert (NUM_DOFS,) in shapes, "the already-flat spelling must be in the grid"

    def test_the_grid_reaches_both_observation_state_readers(self):
        """A ``state_keys`` list on the obs selects convention 1, its absence convention 3.

        Both readers must be exercised; if one branch were unreachable the
        parity assertions would pass while grading a single code path.
        """
        policy = _policy()
        # Convention 3's key list is populated at construction, so the
        # state-keys-absent cells really do land in convention 3.
        assert list(policy._robot_state_keys) == list(JOINT_NAMES)

        # With a *reversed* key list on the obs the two conventions disagree, so
        # the answer names which one ran: convention 1 reads the obs list.
        reversed_keys = list(reversed(JOINT_NAMES))
        state = np.array([POSE[JOINT_NAMES.index(name)] for name in reversed_keys], dtype=np.float32)
        by_convention_1 = _resolved_pose({"observation.state": state, "state_keys": reversed_keys}, policy)
        by_convention_3 = _resolved_pose({"observation.state": state}, _policy())
        assert np.array_equal(by_convention_1, POSE)
        assert not np.array_equal(by_convention_3, POSE), "the two conventions must be distinguishable"


class TestABatchedObservationStateReadsAsTheFlatVector:
    """The regression: one state vector, four spellings, two conventions, one answer."""

    @pytest.mark.parametrize("spelling_name", sorted(_SPELLINGS))
    @pytest.mark.parametrize("with_state_keys", [False, True], ids=["convention_3", "convention_1"])
    def test_every_spelling_and_convention_resolves_the_same_pose(self, spelling_name, with_state_keys):
        observation: dict[str, Any] = {"observation.state": _SPELLINGS[spelling_name]}
        if with_state_keys:
            observation["state_keys"] = list(JOINT_NAMES)
        assert np.array_equal(_resolved_pose(observation), POSE)

    def test_a_batched_state_is_readable_with_the_documented_key_list(self):
        """Supplying ``state_keys`` must not make a readable observation unreadable."""
        batched = POSE.reshape(1, NUM_DOFS)
        without = _resolved_pose({"observation.state": batched})
        with_keys = _resolved_pose({"observation.state": batched, "state_keys": list(JOINT_NAMES)})
        assert np.array_equal(without, with_keys)
        assert np.array_equal(with_keys, POSE)


class TestTheNormalizationIsSingleSourced:
    """``observation.state`` is normalized once, ahead of every convention.

    A per-branch copy is how the two readers drifted apart in the first place,
    so the placement is pinned structurally rather than trusted to review.
    """

    @staticmethod
    def _pack_by_name_ast() -> ast.FunctionDef:
        source = textwrap.dedent(inspect.getsource(ProtoMotionsPolicy._pack_by_name))
        function = ast.parse(source).body[0]
        assert isinstance(function, ast.FunctionDef)
        return function

    def test_the_state_vector_is_assigned_exactly_once(self):
        function = self._pack_by_name_ast()
        assignments = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "state_arr" for t in node.targets)
        ]
        assert len(assignments) == 1, [ast.unparse(a) for a in assignments]
        assert "reshape(-1)" in ast.unparse(assignments[0])

    def test_the_normalization_precedes_every_convention(self):
        function = self._pack_by_name_ast()
        flattening = [
            node for node in ast.walk(function) if isinstance(node, ast.Assign) and "reshape(-1)" in ast.unparse(node)
        ]
        assert len(flattening) == 1, [ast.unparse(node) for node in flattening]
        branches = [node for node in function.body if isinstance(node, ast.If)]
        assert branches, "the conventions are expressed as top-level branches"
        assert flattening[0].lineno < min(branch.lineno for branch in branches)


class TestThePerJointConvention:
    """Convention 2: ``<joint>`` and ``<joint>.vel`` scalars straight off the obs.

    This is the shape a MuJoCo rollout supplies, one scalar per hinge joint.
    """

    @staticmethod
    def _per_joint(pos: np.ndarray, vel: np.ndarray) -> dict[str, Any]:
        obs: dict[str, Any] = {name: float(pos[i]) for i, name in enumerate(JOINT_NAMES)}
        obs.update({f"{name}.vel": float(vel[i]) for i, name in enumerate(JOINT_NAMES)})
        return obs

    def test_per_joint_scalars_resolve_positions_and_velocities(self):
        velocities = (POSE * -2.0).astype(np.float32)
        policy = _policy()
        observation = self._per_joint(POSE, velocities)
        assert np.array_equal(_resolved_pose(observation, policy), POSE)
        # The velocity read is a separate suffix pass over the same dict.
        assert np.array_equal(policy._pack_by_name(observation, ".vel"), velocities)

    def test_a_partially_populated_obs_names_the_missing_joint(self):
        partial = {name: 0.5 for name in JOINT_NAMES[:5]}
        with pytest.raises(KeyError, match=r"missing from obs \(found 5/29\)"):
            _policy()._pack_by_name(partial, "")

    def test_a_null_observation_state_falls_through_to_the_per_joint_keys(self):
        """``observation.state: None`` must not shadow a convention that can answer."""
        observation = self._per_joint(POSE, np.zeros(NUM_DOFS, dtype=np.float32))
        observation["observation.state"] = None
        observation["state_keys"] = list(JOINT_NAMES)
        assert np.array_equal(_resolved_pose(observation), POSE)


class TestTheObsSuppliedKeyListConvention:
    """Convention 1: ``observation.state`` ordered by a ``state_keys`` list on the obs."""

    def test_a_key_list_missing_a_joint_names_the_joint_and_the_list(self):
        short = [name for name in JOINT_NAMES if name != "left_knee_joint"]
        state = np.array([POSE[JOINT_NAMES.index(name)] for name in short], dtype=np.float32)
        with pytest.raises(KeyError, match=r"'left_knee_joint' missing from observation.state's state_keys"):
            _policy()._pack_by_name({"observation.state": state, "state_keys": short}, "")

    def test_a_batched_state_with_a_short_key_list_reports_the_same_refusal(self):
        """The flattening must not turn a missing name into an out-of-range index."""
        short = [name for name in JOINT_NAMES if name != "left_knee_joint"]
        state = np.array([POSE[JOINT_NAMES.index(name)] for name in short], dtype=np.float32)
        with pytest.raises(KeyError, match=r"'left_knee_joint' missing from observation.state's state_keys"):
            _policy()._pack_by_name({"observation.state": state.reshape(1, -1), "state_keys": short}, "")


class TestTheRobotStateKeysConvention:
    """Convention 3: one ``observation.state`` ordered by the runtime's key list."""

    def test_a_reordered_key_list_is_resolved_by_name_not_position(self):
        reordered = list(reversed(JOINT_NAMES))
        policy = _policy()
        policy.set_robot_state_keys(reordered)
        state = np.array([POSE[JOINT_NAMES.index(name)] for name in reordered], dtype=np.float32)
        assert np.array_equal(_resolved_pose({"observation.state": state}, policy), POSE)

    def test_a_batched_reordered_state_resolves_identically(self):
        reordered = list(reversed(JOINT_NAMES))
        policy = _policy()
        policy.set_robot_state_keys(reordered)
        state = np.array([POSE[JOINT_NAMES.index(name)] for name in reordered], dtype=np.float32)
        resolved = _resolved_pose({"observation.state": state.reshape(1, NUM_DOFS)}, policy)
        assert np.array_equal(resolved, POSE)

    def test_a_key_list_naming_no_velocities_names_the_joint_it_cannot_resolve(self):
        """A key list of joint names cannot answer the ``.vel`` pass."""
        policy = _policy()
        with pytest.raises(KeyError, match=r"missing from self\._robot_state_keys"):
            policy._pack_by_name({"observation.state": POSE}, ".vel")


class TestAnObservationNoConventionCanAnswer:
    """The dead end names every convention that could have answered."""

    def test_the_refusal_names_all_three_routes(self):
        with pytest.raises(KeyError) as excinfo:
            _policy()._pack_by_name({}, "")
        message = str(excinfo.value)
        assert "could not resolve dof values with suffix ''" in message
        for route in ("observation.state", "state_keys", "per-joint keys", "dof_pos"):
            assert route in message, route
