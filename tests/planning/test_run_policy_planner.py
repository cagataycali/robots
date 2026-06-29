"""Integration: a planner steers run_policy by varying the locomotion goal kwargs.

Pins the ``run_policy(planner=...)`` contract: each policy query merges the
planner's current command (``target_velocity`` / ``target_height`` /
``locomotion_style``) over the static ``policy_kwargs`` (planner wins), and the
planner's input-thread lifecycle is started before and stopped after the rollout.
A refactor that drops the planner sampling or its lifecycle fails here.
"""

from __future__ import annotations

from typing import Any

from strands_robots.planning import Planner, PlannerCommand
from strands_robots.policies.base import Policy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner


class FakeSim(SimEngine):
    """Minimal physics-free SimEngine for the run loop."""

    def __init__(self, joint_names: tuple[str, ...] = ("j0", "j1")) -> None:
        self._joint_names = list(joint_names)
        self._robots = {"fake_robot": self._joint_names}

    def create_world(self, timestep=None, gravity=None, ground_plane=True):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def destroy(self):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def reset(self):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def step(self, n_steps: int = 1):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def get_state(self):  # type: ignore[no-untyped-def]
        return {"sim_time": 0.0, "step_count": 0}

    def add_robot(self, name, **kw):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def remove_robot(self, name):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._robots.get(robot_name, []))

    def add_object(self, name, **kw):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def remove_object(self, name):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):  # type: ignore[no-untyped-def]
        return {n: 0.0 for n in self._joint_names}

    def send_action(self, action, robot_name=None, n_substeps=1):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def render(self, camera_name="default", width=None, height=None):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}


class _GoalRecordingPolicy(Policy):
    """Records the kwargs of every get_actions call; emits one zero action."""

    def __init__(self) -> None:
        self._keys: list[str] = []
        self.kwargs_seen: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "goal_recording"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = list(robot_state_keys)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.kwargs_seen.append(dict(kwargs))
        return [{k: 0.0 for k in self._keys}]


class _SequencePlanner(Planner):
    """Deterministic planner: each poll() advances through a fixed command list."""

    def __init__(self, commands: list[PlannerCommand]) -> None:
        self._commands = commands
        self._i = 0
        self.started = False
        self.stopped = False
        self.freq: float | None = None

    def set_control_frequency(self, hz: float) -> None:
        self.freq = hz

    def poll(self) -> PlannerCommand:
        cmd = self._commands[min(self._i, len(self._commands) - 1)]
        self._i += 1
        return cmd

    def reset(self) -> None:
        self._i = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    @property
    def provider_name(self) -> str:
        return "sequence"


def _run_with_planner(planner: Planner, policy: Policy, n_steps: int = 4, **kw: Any) -> dict[str, Any]:
    sim = FakeSim()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
    return PolicyRunner(sim).run(
        "fake_robot",
        policy,
        instruction="walk",
        duration=float(n_steps) / 50.0,
        control_frequency=50.0,
        action_horizon=1,
        fast_mode=True,
        planner=planner,
        **kw,
    )


def test_planner_varies_target_velocity_each_query() -> None:
    policy = _GoalRecordingPolicy()
    commands = [
        PlannerCommand(root_vel=(0.1, 0.0, 0.0)),
        PlannerCommand(root_vel=(0.2, 0.0, 0.0)),
        PlannerCommand(root_vel=(0.3, 0.0, 0.1), style="stealth"),
    ]
    _run_with_planner(_SequencePlanner(commands), policy, n_steps=3)
    seen = [k["target_velocity"] for k in policy.kwargs_seen]
    assert seen[0] == [0.1, 0.0, 0.0]
    assert seen[1] == [0.2, 0.0, 0.0]
    assert seen[2] == [0.3, 0.0, 0.1]
    assert policy.kwargs_seen[2]["locomotion_style"] == "stealth"


def test_planner_command_overrides_static_policy_kwargs() -> None:
    policy = _GoalRecordingPolicy()
    planner = _SequencePlanner([PlannerCommand(root_vel=(0.7, 0.0, 0.0))])
    _run_with_planner(planner, policy, n_steps=2, policy_kwargs={"target_velocity": [0.0, 0.0, 0.0], "extra": 1})
    # Planner wins on target_velocity; unrelated static kwargs survive.
    assert policy.kwargs_seen[0]["target_velocity"] == [0.7, 0.0, 0.0]
    assert policy.kwargs_seen[0]["extra"] == 1


def test_no_planner_preserves_static_kwargs() -> None:
    policy = _GoalRecordingPolicy()
    sim = FakeSim()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
    PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=2.0 / 50.0,
        control_frequency=50.0,
        action_horizon=1,
        fast_mode=True,
        policy_kwargs={"target_velocity": [0.5, 0.0, 0.0]},
    )
    assert all(k["target_velocity"] == [0.5, 0.0, 0.0] for k in policy.kwargs_seen)
    assert all("locomotion_style" not in k for k in policy.kwargs_seen)


def test_run_policy_starts_and_stops_planner_lifecycle() -> None:
    sim = FakeSim()
    policy = _GoalRecordingPolicy()
    planner = _SequencePlanner([PlannerCommand(root_vel=(0.4, 0.0, 0.0))])
    sim.run_policy(
        robot_name="fake_robot",
        policy_object=policy,
        duration=2.0 / 50.0,
        control_frequency=50.0,
        action_horizon=1,
        fast_mode=True,
        planner=planner,
        wbc_install_torque_control=False,
    )
    assert planner.started
    assert planner.stopped
    assert planner.freq == 50.0
    assert policy.kwargs_seen[0]["target_velocity"] == [0.4, 0.0, 0.0]
