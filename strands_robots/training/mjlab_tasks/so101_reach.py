"""SO-101 reach: drive the ``gripper`` site to a random target in the workspace.

Defined on the same MJCF the MuJoCo/mjlab backends load
(``robotstudio_so101/so101_new_calib.xml``), so a policy trained here runs on
``Robot("so101")`` in either engine without any retargeting. Observation
layout (documented because the ONNX provider rebuilds it from raw state):

``joint_pos_rel(6) | joint_vel_rel(6) | ee_to_target(3) | last_action(6)`` = 21.

Actions are joint-position deltas around the default pose (``scale``, as in
mjlab's ``JointPositionActionCfg(use_default_offset=True)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

TASK_ID = "Strands-Reach-SO101"
EE_SITE = "gripper"
ACTION_SCALE = 0.25
SUCCESS_M = 0.03
# Reachable box in the base frame (arm reach ~0.35 m; targets above the table).
TARGET_X = (0.12, 0.30)
TARGET_Y = (-0.20, 0.20)
TARGET_Z = (0.08, 0.30)


# ------------------------------------------------------------------ command


class ReachCommand(CommandTerm):
    """A target point in the robot base frame, resampled every few seconds."""

    cfg: ReachCommandCfg

    def __init__(self, cfg: ReachCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        self.target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos_b

    def target_pos_w(self) -> torch.Tensor:
        base_pos = self.robot.data.root_link_pos_w
        base_quat = self.robot.data.root_link_quat_w
        return base_pos + quat_apply(base_quat, self.target_pos_b)

    def ee_pos_w(self) -> torch.Tensor:
        if not hasattr(self, "_ee_site_id"):
            self._ee_site_id = self.robot.find_sites(EE_SITE)[0][0]
        return self.robot.data.site_pos_w[:, self._ee_site_id]

    def _update_metrics(self) -> None:
        err = torch.norm(self.target_pos_w() - self.ee_pos_w(), dim=-1)
        self.metrics["position_error"] = err
        self.metrics["at_goal"] = (err < self.cfg.success_threshold).float()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        r = self.cfg.range
        t = torch.empty(n, 3, device=self.device)
        t[:, 0].uniform_(*r.x)
        t[:, 1].uniform_(*r.y)
        t[:, 2].uniform_(*r.z)
        self.target_pos_b[env_ids] = t

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        return

    def _debug_vis_impl(self, visualizer) -> None:  # pragma: no cover - viewer only
        for pos in self.target_pos_w().cpu().numpy():
            visualizer.add_sphere(pos, radius=0.015, color=(1.0, 0.5, 0.0, 0.5))


@dataclass(kw_only=True)
class ReachCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    success_threshold: float = SUCCESS_M

    @dataclass
    class RangeCfg:
        x: tuple[float, float] = TARGET_X
        y: tuple[float, float] = TARGET_Y
        z: tuple[float, float] = TARGET_Z

    range: RangeCfg = field(default_factory=RangeCfg)

    def build(self, env: ManagerBasedRlEnv) -> ReachCommand:
        return ReachCommand(self, env)


# --------------------------------------------------------------- mdp terms


def ee_to_target(env: ManagerBasedRlEnv, command_name: str = "reach") -> torch.Tensor:
    """Vector from the gripper site to the target, in the base frame."""
    cmd: ReachCommand = env.command_manager.get_term(command_name)
    vec_w = cmd.target_pos_w() - cmd.ee_pos_w()
    return quat_apply(quat_inv(cmd.robot.data.root_link_quat_w), vec_w)


def reach_reward(env: ManagerBasedRlEnv, std: float, command_name: str = "reach") -> torch.Tensor:
    cmd: ReachCommand = env.command_manager.get_term(command_name)
    err = torch.norm(cmd.target_pos_w() - cmd.ee_pos_w(), dim=-1)
    return torch.exp(-((err / std) ** 2))


def reach_success(env: ManagerBasedRlEnv, command_name: str = "reach") -> torch.Tensor:
    cmd: ReachCommand = env.command_manager.get_term(command_name)
    return cmd.metrics["at_goal"]


# ---------------------------------------------------------------- env cfg


def so101_entity_cfg():
    """The so101 EntityCfg the MjlabEngine builds, reused here verbatim."""
    from strands_robots.assets import resolve_model_path, resolve_robot_name
    from strands_robots.simulation.mjlab.simulation import MjlabEngine, _RobotSpec

    path = str(resolve_model_path(resolve_robot_name("so101")))
    spec = _RobotSpec(
        name="robot",
        path=path,
        position=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        keyframe=None,
        actuator_names=["1", "2", "3", "4", "5", "6"],
    )
    return MjlabEngine._robot_entity_cfg(None, spec)  # type: ignore[arg-type]


def so101_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The reach task: 21-dim actor obs, 6 joint-position actions, 50 Hz control."""
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)),
        "ee_to_target": ObservationTermCfg(func=ee_to_target, noise=Unoise(n_min=-0.005, n_max=0.005)),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg({**actor_terms}, enable_corruption=not play),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",), scale=ACTION_SCALE, use_default_offset=True
        )
    }
    commands = {"reach": ReachCommandCfg(resampling_time_range=(3.0, 5.0), debug_vis=play)}
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {}, "velocity_range": {}}
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.3, 0.3),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }
    rewards = {
        "reach_coarse": RewardTermCfg(func=reach_reward, weight=1.0, params={"std": 0.15}),
        "reach_fine": RewardTermCfg(func=reach_reward, weight=2.0, params={"std": 0.03}),
        "success": RewardTermCfg(func=reach_success, weight=1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}
        ),
    }
    terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=1.0,
            entities={"robot": so101_entity_cfg()},
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base",
            distance=1.0,
            elevation=-15.0,
            azimuth=135.0,
        ),
        sim=SimulationCfg(
            nconmax=20,
            njmax=200,
            mujoco=MujocoCfg(timestep=0.002, iterations=10, ls_iterations=20, cone="elliptic"),
        ),
        decimation=10,  # 50 Hz control, the rate Robot("so101") policies run at
        episode_length_s=8.0 if not play else 20.0,
    )


def so101_reach_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO settings sized for a 6-DoF reach (300 iterations at 1024 envs is enough)."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.98,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="so101_reach",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=300,
    )


def register() -> str:
    """Register the task with mjlab once; returns its id."""
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    if TASK_ID not in list_tasks():
        register_mjlab_task(
            task_id=TASK_ID,
            env_cfg=so101_reach_env_cfg(),
            play_env_cfg=so101_reach_env_cfg(play=True),
            rl_cfg=so101_reach_ppo_runner_cfg(),
        )
    return TASK_ID


__all__ = ["TASK_ID", "ACTION_SCALE", "register", "so101_reach_env_cfg", "so101_reach_ppo_runner_cfg"]
