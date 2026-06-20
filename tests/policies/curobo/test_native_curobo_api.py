"""Coverage for :class:`CuroboPolicy`'s native cuRobo-API construction paths.

The smoke tests in ``test_policy.py`` inject a pre-built planner via
``motion_gen=`` and therefore never touch the lazy cuRobo imports inside
:meth:`CuroboPolicy._build_motion_gen`, :meth:`_build_start_state`,
:meth:`_build_goal_pose`, :meth:`_build_goal_joint_state`,
:meth:`_planner_tensor_kwargs`, and the tensor branch of
:meth:`_extract_trajectory`. Those branches only run when ``import curobo``
succeeds - i.e. on a GPU box with cuRobo installed - so they are normally
exercised only by the live integration test.

This module installs a lightweight fake ``curobo`` package into
``sys.modules`` and lets the policy run against real (CPU) torch tensors.
That exercises the production code path that builds cuRobo's native
``DeviceCfg`` / ``JointState`` / ``GoalToolPose`` types and collapses the
``[batch, horizon, T, ndof]`` interpolated-plan tensor down to a flat
``[T, ndof]`` trajectory - on any machine, no CUDA required.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

import strands_robots.utils as sr_utils
from strands_robots.policies.curobo import CuroboPolicy

# ---------------------------------------------------------------------------
# Fake cuRobo package (real torch under the hood)
# ---------------------------------------------------------------------------


class _FakeDeviceCfg:
    """Stand-in for ``curobo.types.DeviceCfg``."""

    def __init__(self, device: object = None, dtype: object = None) -> None:
        self.device = device
        self.dtype = dtype


class _FakeJointState:
    """Stand-in for ``curobo.types.JointState`` (records construction)."""

    def __init__(self, position: object, joint_names: object = None) -> None:
        self.position = position
        self.joint_names = joint_names

    @classmethod
    def from_position(cls, position: object, joint_names: object = None) -> _FakeJointState:
        return cls(position, joint_names)


class _FakeGoalToolPose:
    """Stand-in for ``curobo.types.GoalToolPose`` (records construction)."""

    def __init__(self, tool_frames: object, position: object, quaternion: object) -> None:
        self.tool_frames = tool_frames
        self.position = position
        self.quaternion = quaternion


class _FakePose:
    """Stand-in for ``curobo.types.Pose`` (unused by the planning path)."""


class _FakeInterpolatedPlan:
    """Mirrors cuRobo's interpolated-plan object: ``.position`` is a tensor."""

    def __init__(self, position: torch.Tensor) -> None:
        self.position = position


class _FakeNativeResult:
    """Native ``plan_pose`` / ``plan_js`` result (tensor trajectory path).

    Unlike the stub result in ``test_policy.py``, this exposes no
    ``trajectory`` list - it forces ``_extract_trajectory`` through the
    ``get_interpolated_plan().position`` tensor branch.
    """

    def __init__(self, position: torch.Tensor, success: bool = True) -> None:
        self.success = success
        self._plan = _FakeInterpolatedPlan(position)

    def get_interpolated_plan(self) -> _FakeInterpolatedPlan:
        return self._plan


class _FakeKinematics:
    def __init__(self) -> None:
        self.tool_frames = ["panda_hand"]


class _FakeMotionPlanner:
    """Native ``MotionPlanner`` stand-in returning real torch tensors."""

    def __init__(self, cfg: object) -> None:
        self.cfg = cfg
        self.kinematics = _FakeKinematics()
        # The policy reads device/dtype off ``.device_cfg``.
        self.device_cfg = _FakeDeviceCfg(device=torch.device("cpu"), dtype=torch.float32)
        self.warmup_called = 0
        self.plan_pose_calls: list[tuple] = []
        self.plan_js_calls: list[tuple] = []
        self.scene_updates: list[object] = []

    def warmup(self) -> None:
        self.warmup_called += 1

    def update_scene(self, world: object) -> None:
        self.scene_updates.append(world)

    def _traj(self, ndof: int) -> torch.Tensor:
        # Native shape: [batch=1, horizon=1, T=4, ndof].
        rows = [[(t + 1) * 0.01 * (i + 1) for i in range(ndof)] for t in range(4)]
        return torch.tensor(rows, dtype=torch.float32).reshape(1, 1, 4, ndof)

    def plan_pose(self, goal: object, start: object) -> _FakeNativeResult:
        self.plan_pose_calls.append((goal, start))
        return _FakeNativeResult(self._traj(7))

    def plan_js(self, start: object, goal: object) -> _FakeNativeResult:
        self.plan_js_calls.append((start, goal))
        return _FakeNativeResult(self._traj(2))


class _FakeMotionPlannerCfg:
    last_kwargs: dict | None = None

    @classmethod
    def create(cls, **kwargs: object) -> _FakeMotionPlannerCfg:
        cls.last_kwargs = dict(kwargs)
        return cls()


@pytest.fixture
def fake_curobo(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``curobo`` package so the native import paths run.

    Yields the ``_FakeMotionPlannerCfg`` class so tests can assert what was
    forwarded into ``MotionPlannerCfg.create``.
    """
    motion_planner_mod = types.ModuleType("curobo.motion_planner")
    setattr(motion_planner_mod, "MotionPlanner", _FakeMotionPlanner)
    setattr(motion_planner_mod, "MotionPlannerCfg", _FakeMotionPlannerCfg)

    types_mod = types.ModuleType("curobo.types")
    setattr(types_mod, "DeviceCfg", _FakeDeviceCfg)
    setattr(types_mod, "JointState", _FakeJointState)
    setattr(types_mod, "GoalToolPose", _FakeGoalToolPose)
    setattr(types_mod, "Pose", _FakePose)

    curobo_pkg = types.ModuleType("curobo")
    setattr(curobo_pkg, "motion_planner", motion_planner_mod)
    setattr(curobo_pkg, "types", types_mod)

    monkeypatch.setitem(sys.modules, "curobo", curobo_pkg)
    monkeypatch.setitem(sys.modules, "curobo.motion_planner", motion_planner_mod)
    monkeypatch.setitem(sys.modules, "curobo.types", types_mod)
    # ``require_optional`` memoises imported modules; clear so it re-imports
    # the fake and the teardown leaves no dangling reference.
    monkeypatch.setitem(sr_utils._lazy_modules, "curobo", curobo_pkg)
    _FakeMotionPlannerCfg.last_kwargs = None
    yield _FakeMotionPlannerCfg
    sr_utils._lazy_modules.pop("curobo", None)


# ---------------------------------------------------------------------------
# Construction via the native MotionPlannerCfg.create path
# ---------------------------------------------------------------------------


class TestNativeConstruction:
    def test_builds_planner_from_robot_config(self, fake_curobo) -> None:
        """robot_config (no stub) builds a planner via MotionPlannerCfg.create."""
        policy = CuroboPolicy(robot_config="franka.yml", warmup=False)
        assert isinstance(policy._motion_planner, _FakeMotionPlanner)
        assert fake_curobo.last_kwargs is not None
        assert fake_curobo.last_kwargs["robot"] == "franka.yml"
        # Free-space planning: no scene_model forwarded.
        assert "scene_model" not in fake_curobo.last_kwargs

    def test_world_config_forwarded_as_scene_model(self, fake_curobo) -> None:
        world = {"cuboid": {"table": {"dims": [1, 1, 0.1], "pose": [0, 0, 0, 1, 0, 0, 0]}}}
        CuroboPolicy(robot_config="ur5e.yml", world_config=world, warmup=False)
        assert fake_curobo.last_kwargs["scene_model"] == world

    def test_extra_planner_kwargs_forwarded(self, fake_curobo) -> None:
        CuroboPolicy(
            robot_config="franka.yml",
            motion_planner_kwargs={"interpolation_dt": 0.02},
            warmup=False,
        )
        assert fake_curobo.last_kwargs["interpolation_dt"] == 0.02

    def test_warmup_invoked_when_building_planner(self, fake_curobo) -> None:
        policy = CuroboPolicy(robot_config="franka.yml", warmup=True)
        assert policy._motion_planner.warmup_called == 1


# ---------------------------------------------------------------------------
# Device-config resolution (native DeviceCfg)
# ---------------------------------------------------------------------------


class TestResolveDeviceCfg:
    def test_none_defaults_to_cpu_when_cuda_unavailable(self) -> None:
        dc = CuroboPolicy._resolve_device_cfg(None, _FakeDeviceCfg, torch)
        assert isinstance(dc, _FakeDeviceCfg)
        # On the CPU test host CUDA is unavailable -> cpu device.
        if not torch.cuda.is_available():
            assert dc.device == torch.device("cpu")

    def test_string_coerced_to_device(self) -> None:
        dc = CuroboPolicy._resolve_device_cfg("cpu", _FakeDeviceCfg, torch)
        assert dc.device == torch.device("cpu")

    def test_torch_device_wrapped(self) -> None:
        dev = torch.device("cpu")
        dc = CuroboPolicy._resolve_device_cfg(dev, _FakeDeviceCfg, torch)
        assert dc.device == dev

    def test_prebuilt_device_cfg_passed_through(self) -> None:
        prebuilt = _FakeDeviceCfg(device=torch.device("cpu"))
        dc = CuroboPolicy._resolve_device_cfg(prebuilt, _FakeDeviceCfg, torch)
        assert dc is prebuilt


# ---------------------------------------------------------------------------
# Cartesian goal -> plan_pose -> tensor trajectory extraction
# ---------------------------------------------------------------------------


class TestNativeCartesianPlan:
    def test_target_pose_builds_goaltoolpose_and_extracts_tensor(self, fake_curobo) -> None:
        policy = CuroboPolicy(robot_config="franka.yml", action_horizon=8, warmup=False)
        actions = policy.get_actions_sync(
            observation_dict={"observation.state": [0.0] * 7},
            instruction="",
            target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
        )
        planner = policy._motion_planner
        # plan_pose received a GoalToolPose built from the native tensors.
        assert len(planner.plan_pose_calls) == 1
        goal, start = planner.plan_pose_calls[0]
        assert isinstance(goal, _FakeGoalToolPose)
        assert goal.tool_frames == ["panda_hand"]
        # Position tensor has the native 5D shape [B,H,L,G,3].
        assert tuple(goal.position.shape) == (1, 1, 1, 1, 3)  # type: ignore[attr-defined]
        assert tuple(goal.quaternion.shape) == (1, 1, 1, 1, 4)  # type: ignore[attr-defined]
        # Start state was built as a native JointState from the observation.
        assert isinstance(start, _FakeJointState)
        # The [1,1,4,7] tensor collapsed to 4 rows of 7 floats.
        assert len(actions) == 4
        assert len(actions[0]) == 7

    def test_default_joint_keys_when_state_keys_unset(self, fake_curobo) -> None:
        policy = CuroboPolicy(robot_config="franka.yml", action_horizon=8, warmup=False)
        actions = policy.get_actions_sync(
            observation_dict={"observation.state": [0.0] * 7},
            instruction="",
            target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
        )
        assert set(actions[0].keys()) == {f"joint_{i}" for i in range(7)}


# ---------------------------------------------------------------------------
# Joint-space goal -> plan_js -> tensor trajectory extraction
# ---------------------------------------------------------------------------


class TestNativeJointSpacePlan:
    def test_target_joints_builds_jointstate_and_routes_to_plan_js(self, fake_curobo) -> None:
        policy = CuroboPolicy(robot_config="franka.yml", action_horizon=8, warmup=False)
        actions = policy.get_actions_sync(
            observation_dict={"observation.state": [0.1, 0.2]},
            instruction="",
            target_joints={"j1": 0.3, "j2": 0.4},
        )
        planner = policy._motion_planner
        assert len(planner.plan_js_calls) == 1
        start, goal_js = planner.plan_js_calls[0]
        assert isinstance(start, _FakeJointState)
        assert isinstance(goal_js, _FakeJointState)
        # Joint names are forwarded onto the goal JointState (sorted by default).
        assert goal_js.joint_names == ["j1", "j2"]
        assert len(actions) == 4
        assert len(actions[0]) == 2

    def test_robot_state_keys_order_used_for_goal_joints(self, fake_curobo) -> None:
        policy = CuroboPolicy(robot_config="franka.yml", action_horizon=8, warmup=False)
        policy.set_robot_state_keys(["j2", "j1"])
        policy.get_actions_sync(
            observation_dict={"observation.state": [0.1, 0.2]},
            instruction="",
            target_joints={"j1": 0.3, "j2": 0.4},
        )
        _start, goal_js = policy._motion_planner.plan_js_calls[0]
        # Honours the configured key order rather than sorting.
        assert goal_js.joint_names == ["j2", "j1"]


# ---------------------------------------------------------------------------
# Per-call collision-scene refresh via native update_scene
# ---------------------------------------------------------------------------


class TestNativeWorldUpdate:
    def test_world_update_routed_to_update_scene(self, fake_curobo) -> None:
        policy = CuroboPolicy(robot_config="franka.yml", action_horizon=8, warmup=False)
        refresh = {"cuboid": {"wall": {"dims": [0.1, 1, 1], "pose": [1, 0, 0, 1, 0, 0, 0]}}}
        policy.get_actions_sync(
            observation_dict={"observation.state": [0.0] * 7},
            instruction="",
            target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            world_update=refresh,
        )
        assert policy._motion_planner.scene_updates == [refresh]
