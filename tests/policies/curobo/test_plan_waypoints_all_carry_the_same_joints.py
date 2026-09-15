"""Every waypoint of a cached cuRobo plan carries the same joints as the first.

:meth:`CuroboPolicy._next_chunk` resolves the joint key names *once* per chunk,
from the width of that chunk's first waypoint, and then pairs those keys with
every waypoint in the chunk. Those keys are a claim about one waypoint applied to
all of them, and the pairing used to be ``strict=False`` - so a waypoint the
claim did not describe was commanded partially instead of reported.

Measured through the public :meth:`~CuroboPolicy.get_actions_sync` against the
stub-planner seam ``_extract_trajectory`` documents, on a 7-joint plan whose
waypoints 12..23 carry only 4 positions:

=========================================== =========================== ==========
plan                                        pre-fix                     post-fix
=========================================== =========================== ==========
rectangular (control)                       24 waypoints, 7 keys each   unchanged
waypoints 12..23 carry 4 of 7 positions     24 waypoints "served"       refused
every waypoint carries no position           3 waypoints, ``{}`` each    refused
=========================================== =========================== ==========

Driven into MuJoCo physics, the ragged plan pre-fix moved the arm 0.15 rad over
the window the plan asked 1.17 rad of, and ended 1.17 rad from the plan's
endpoint on joint 1 where the healthy plan ended 0.15 rad from it - reported as
a successful plan, with no refusal and no warning from the policy.

Worse, *which* silent wrong thing happened depended on ``action_horizon``, which
is a streaming parameter and no part of the plan. For that same corrupt plan:

* the narrow waypoints landed at the head of a chunk (``action_horizon`` 1, 4, 5)
  - so ``_resolve_joint_keys(4)`` found no 4-element ``set_robot_state_keys``
  match and fell back to positional ``joint_0..joint_3`` labels, which resolve to
  no actuator at all; the simulator dropped every value.
* the narrow waypoints landed *inside* a chunk whose first waypoint was full
  (``action_horizon`` 16) - so they were commanded on the real joint keys with
  only 4 of the 7 filled, leaving three joints holding mid-motion.

A planner's degree-of-freedom count does not change mid-plan, so a plan that is
not rectangular is a broken plan. It is therefore graded when it is *cached*,
beside the ``_MAX_TRAJECTORY_WAYPOINTS`` guard, rather than when a chunk of it is
served: the offending waypoint can sit in the second or the tenth chunk, and
refusing at serve time would refuse after the arm had already run the first.
:meth:`MoveIt2Policy._unpack_trajectory` refuses a positionless waypoint for the
same reason, and ``docs/policies/moveit2.md`` states the rule the two planner
policies now share - a plan that commands nothing is a planning failure, not a
successful no-op plan.

All contracts run against lightweight stubs - no GPU, no cuRobo install.
"""

from __future__ import annotations

import types

import pytest

from strands_robots.policies.curobo import CuroboPolicy
from strands_robots.policies.curobo.policy import _trajectory_shape_error

NDOF = 7
STATE_KEYS = [f"panda_joint{i + 1}" for i in range(NDOF)]
HORIZONS = (1, 4, 5, 16)


def _plan(n_waypoints: int = 12, width: int = NDOF) -> list[list[float]]:
    """A rectangular plan: ``n_waypoints`` rows of ``width`` distinct positions."""
    return [[round(0.01 * (t + 1) * (i + 1), 4) for i in range(width)] for t in range(n_waypoints)]


class _StubResult:
    """A plan result exposing the ``trajectory`` list seam ``_extract_trajectory`` documents."""

    def __init__(self, trajectory: list[list[float]]) -> None:
        self.success = True
        self.status = "ok"
        self.trajectory = trajectory


class _StubPlanner:
    """Joint-space-only planner returning a caller-chosen trajectory."""

    def __init__(self, trajectory: list[list[float]]) -> None:
        self._trajectory = trajectory
        self.kinematics = types.SimpleNamespace(tool_frames=["tool0"])

    def plan_single_js(self, start: object, goal: object) -> _StubResult:
        return _StubResult(self._trajectory)


def _policy(trajectory: list[list[float]], action_horizon: int = 16) -> CuroboPolicy:
    policy = CuroboPolicy(motion_gen=_StubPlanner(trajectory), action_horizon=action_horizon, warmup=False)
    policy.set_robot_state_keys(STATE_KEYS)
    return policy


def _stream(policy: CuroboPolicy, n_waypoints: int) -> list[dict[str, float]]:
    """Drain the cached plan the way the execution loop does, chunk by chunk."""
    served: list[dict[str, float]] = []
    while len(served) < n_waypoints:
        actions = policy.get_actions_sync(
            observation_dict={"observation.state": [0.0] * NDOF},
            instruction="",
            target_joints={"joint1": 0.1},
        )
        if not actions:
            break
        served.extend(actions)
    return served


class TestPremises:
    """The stubs really drive the code under test, so the refusals are not vacuous."""

    def test_a_rectangular_plan_commands_every_joint_at_every_waypoint(self) -> None:
        """The control: nothing about this change touches a well-formed plan."""
        served = _stream(_policy(_plan(12)), 12)
        assert len(served) == 12
        assert all(sorted(action) == sorted(STATE_KEYS) for action in served)
        # The values are the planned ones, on the caller's own key names.
        assert served[0] == dict(zip(STATE_KEYS, _plan(12)[0], strict=True))

    def test_a_narrow_waypoint_would_have_been_keyed_by_the_first_waypoints_width(self) -> None:
        """``_resolve_joint_keys`` has no 4-element match, so it falls back to positional labels.

        This is the mechanism the refusal exists for: those labels name no
        actuator, so the values reached the robot and were dropped there.
        """
        policy = _policy(_plan(1))
        assert policy._resolve_joint_keys(NDOF) == STATE_KEYS
        assert policy._resolve_joint_keys(4) == ["joint_0", "joint_1", "joint_2", "joint_3"]


class TestARaggedPlanIsRefusedWhenItIsCached:
    """A waypoint the first waypoint does not describe is refused, not commanded partially."""

    @pytest.mark.parametrize("width", [1, 4, NDOF - 1])
    def test_a_narrower_waypoint_is_refused_naming_both_widths(self, width: int) -> None:
        plan = _plan(12)
        plan[5] = plan[5][:width]
        with pytest.raises(RuntimeError) as excinfo:
            _stream(_policy(plan), 12)
        message = str(excinfo.value)
        assert "waypoint 5 of 12" in message
        assert f"carries {width} joint positions but waypoint 0 carries {NDOF}" in message

    def test_a_wider_waypoint_is_refused_rather_than_having_its_tail_dropped(self) -> None:
        """The narrow case loses joints; the wide case loses positions. Both are refused."""
        plan = _plan(12)
        plan[7] = plan[7] + [9.9, 9.9]
        with pytest.raises(RuntimeError) as excinfo:
            _stream(_policy(plan), 12)
        assert f"waypoint 7 of 12 carries {NDOF + 2} joint positions but waypoint 0 carries {NDOF}" in str(
            excinfo.value
        )

    @pytest.mark.parametrize("index", [0, 6, 11])
    def test_a_positionless_waypoint_is_refused_wherever_it_sits(self, index: int) -> None:
        """An empty waypoint unpacks to ``{}`` - a command inside a chunk that moves no joint."""
        plan = _plan(12)
        plan[index] = []
        with pytest.raises(RuntimeError) as excinfo:
            _stream(_policy(plan), 12)
        message = str(excinfo.value)
        assert f"waypoint {index} of 12 carries no joint position" in message
        assert "commands nothing" in message

    def test_a_plan_of_only_positionless_waypoints_is_refused(self) -> None:
        """Pre-fix this returned ``[{}, {}, {}]`` - truthy, so every ``if not actions`` guard passed."""
        with pytest.raises(RuntimeError, match="waypoint 0 of 3 carries no joint position"):
            _stream(_policy([[], [], []]), 3)

    def test_the_refusal_names_the_goal_and_says_nothing_was_cached(self) -> None:
        plan = _plan(12)
        plan[5] = []
        with pytest.raises(RuntimeError) as excinfo:
            _stream(_policy(plan), 12)
        message = str(excinfo.value)
        assert message.startswith("CuroboPolicy planning failed:")
        assert "target_joints={'joint1': 0.1}" in message
        assert "Refusing to cache." in message

    def test_nothing_of_a_refused_plan_is_left_cached(self) -> None:
        """The refusal has to leave no waypoint behind, or the next call would serve it."""
        plan = _plan(12)
        plan[5] = plan[5][:4]
        policy = _policy(plan)
        with pytest.raises(RuntimeError):
            _stream(policy, 12)
        assert policy._cached_trajectory == []
        assert policy._next_chunk() == []


class TestTheVerdictDoesNotDependOnTheStreamingHorizon:
    """``action_horizon`` decides chunk boundaries; it must not decide correctness."""

    @pytest.mark.parametrize("action_horizon", HORIZONS)
    def test_one_ragged_plan_is_refused_at_every_horizon(self, action_horizon: int) -> None:
        """Pre-fix this plan produced two *different* silent wrong behaviours across these four.

        With the narrow waypoints at a chunk head (horizons 1, 4, 5) they were
        keyed ``joint_0..joint_3`` and reached no actuator; inside a chunk whose
        first waypoint was full (horizon 16) they were keyed correctly but only
        4 of 7 filled. Both were reported as a successful plan.
        """
        plan = _plan(12)
        for index in range(5, 12):
            plan[index] = plan[index][:4]
        with pytest.raises(RuntimeError, match="waypoint 5 of 12 carries 4 joint positions"):
            _stream(_policy(plan, action_horizon=action_horizon), 12)

    @pytest.mark.parametrize("action_horizon", HORIZONS)
    def test_a_rectangular_plan_is_served_whole_at_every_horizon(self, action_horizon: int) -> None:
        """The control arm of the same table: the horizon still only sets the chunk width."""
        served = _stream(_policy(_plan(12), action_horizon=action_horizon), 12)
        assert len(served) == 12
        assert all(sorted(action) == sorted(STATE_KEYS) for action in served)


class TestThePairingItselfEnforcesTheInvariant:
    """The keys-to-positions pairing is the second door, behind the cache-time grading.

    With every cached plan graded, ``_next_chunk`` pairs equal-width sequences by
    construction - so pairing them strictly costs nothing and is unreachable
    through :meth:`~CuroboPolicy.get_actions_sync`. It is what stops a future code
    path that fills ``_cached_trajectory`` without grading it from reintroducing a
    silently partial command: the pairing refuses instead of truncating.
    """

    def test_a_waypoint_narrower_than_its_chunks_keys_is_refused_not_truncated(self) -> None:
        policy = _policy(_plan(1))
        policy._cached_trajectory = [_plan(1)[0], _plan(1)[0][:4]]
        policy._cached_cursor = 0
        with pytest.raises(ValueError, match="zip"):
            policy._next_chunk()

    def test_an_equal_width_chunk_is_still_paired_whole(self) -> None:
        """The control: strict pairing does not change a well-formed chunk."""
        policy = _policy(_plan(1))
        policy._cached_trajectory = _plan(2)
        policy._cached_cursor = 0
        chunk = policy._next_chunk()
        assert len(chunk) == 2
        assert all(sorted(action) == sorted(STATE_KEYS) for action in chunk)


class TestTrajectoryShapeErrorDomain:
    """The grading helper on its own: what it refuses, and what is not its subject."""

    @pytest.mark.parametrize("width", [1, 2, 6, 7, 9])
    def test_a_rectangular_plan_of_any_width_is_accepted(self, width: int) -> None:
        assert _trajectory_shape_error(_plan(5, width=width)) is None

    def test_an_empty_plan_is_not_this_helpers_subject(self) -> None:
        """A plan with no waypoint at all is graded by the caller's own emptiness handling."""
        assert _trajectory_shape_error([]) is None

    def test_the_first_offending_waypoint_is_the_one_reported(self) -> None:
        plan = _plan(12)
        plan[9] = []
        plan[4] = plan[4][:3]
        error = _trajectory_shape_error(plan)
        assert error is not None
        assert "waypoint 4 of 12" in error
        assert "waypoint 9" not in error

    def test_a_positionless_first_waypoint_is_reported_as_positionless_not_as_a_width_mismatch(self) -> None:
        """Width 0 is the degenerate reference width, so it needs its own reason."""
        plan = _plan(3)
        plan[0] = []
        error = _trajectory_shape_error(plan)
        assert error is not None
        assert "waypoint 0 of 3 carries no joint position" in error
