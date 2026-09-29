"""Humanoid beyond flat velocity: G1 on rough terrain, and a get-up task composed from reward terms.

Two tasks on the Unitree G1 through the mjlab trainer:

* ``rough``: mjlab's own ``Mjlab-Velocity-Rough-Unitree-G1`` (procedural terrain
  generator with a difficulty curriculum, 187-ray height scan in the actor).
* ``getup``: ``Mjlab-Velocity-Flat-Unitree-G1`` re-purposed in under 100 lines:
  the robot is dropped supine or prone, the tracking rewards are replaced by a
  height + upright shaping, the fall termination is removed, so the only way
  to be rewarded is to get up and stand.

Stages (each a subcommand so a long training can run under nohup and be polled)::

    python examples/mjlab/03_humanoid_beyond_velocity.py train --task rough --num-envs 2048 --iterations 1500 --run-dir runs/g1_rough
    python examples/mjlab/03_humanoid_beyond_velocity.py export --task rough --checkpoint runs/g1_rough/model_1499.pt --onnx runs/g1_rough.onnx
    python examples/mjlab/03_humanoid_beyond_velocity.py eval-native --task rough --onnx runs/g1_rough.onnx --out native.json
    python examples/mjlab/03_humanoid_beyond_velocity.py eval-s2s --onnx runs/g1_rough.onnx --out s2s.json

``eval-native`` plays the ONNX actor inside mjlab's play environment (rough
terrain at full difficulty for ``rough``; supine/prone drops for ``getup``) with
one world per command, and reports survival / tracking error / final height.
``eval-s2s`` plays the same ONNX on the classic MuJoCo backend (flat plane) with
the deep lane's 10 s survive-on-4-commands harness; the height-scan term the
provider does not know is supplied by a subclass (flat plane: every ray hits
z = 0, so each height is the pelvis height).

Install (one step; the lerobot extra is for the recorders)::

    uv pip install "strands-robots[lerobot,sim-mjlab]"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROBOT = "unitree_g1"
TASKS = {"rough": "Mjlab-Velocity-Rough-Unitree-G1", "getup": "Mjlab-Velocity-Flat-Unitree-G1"}
COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "fwd_0.5": (0.5, 0.0, 0.0),
    "fwd_1.0": (1.0, 0.0, 0.0),
    "yaw_0.5": (0.0, 0.0, 0.5),
}
STAND_Z = 0.72  # pelvis height of a standing G1 (deep lane native harness: 0.755-0.764)
# Get-up reward shapes. v1 (std 0.3) pays 66 percent of the height term for a 0.525 m crouch and PPO
# settled there (0/4 stood up at 1000 iterations); v2 sharpens the height bell, raises the standing bonus
# and the upright weight so a crouch is worth 18 percent and only an upright stand collects the rest.
GETUP_REWARDS = {
    "v1": {"height_std": 0.3, "height_w": 2.0, "standing_w": 3.0, "standing_dz": 0.1, "upright_w": 2.0},
    "v2": {"height_std": 0.15, "height_w": 2.0, "standing_w": 5.0, "standing_dz": 0.15, "upright_w": 3.0},
    # v3: F21 read at v2 checkpoint 600 shows the crouch keeps torso_link upright (the stock ``upright`` term
    # rewards the torso) while the pelvis stays pitched, which ``standing`` gates on with a hard AND. A shaped
    # pelvis-upright term gives PPO the gradient the gate withholds.
    "v3": {
        "height_std": 0.15,
        "height_w": 2.0,
        "standing_w": 5.0,
        "standing_dz": 0.15,
        "upright_w": 3.0,
        "pelvis_upright_w": 2.0,
        "pelvis_upright_std": 0.4,
    },
}
GETUP_REWARD = os.environ.get("GETUP_REWARD", "v1")
FALL_Z = 0.35


# ------------------------------------------------------------------ getup


def getup_env_cfg(play: bool = False):
    """Flat G1 velocity cfg turned into a get-up task (the <100 lines the brief asked for)."""
    import torch
    from mjlab.envs.mdp import events as envs_mdp
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(TASKS["getup"], play=play)

    def pelvis_height(env, asset_cfg=SceneEntityCfg("robot")):
        return env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]

    def height_reward(env, target: float, std: float, asset_cfg=SceneEntityCfg("robot")):
        z = pelvis_height(env, asset_cfg)
        return torch.exp(-torch.square(torch.clamp(target - z, min=0.0)) / std**2)

    def standing(env, target: float, asset_cfg=SceneEntityCfg("robot")):
        """1 when the pelvis is above ``target`` and the torso is within 20 deg of upright."""
        asset = env.scene[asset_cfg.name]
        g = asset.data.projected_gravity_b
        up = g[:, 2] < -math.cos(math.radians(20.0))
        return (pelvis_height(env, asset_cfg) > target).float() * up.float()

    def pelvis_upright(env, std: float, asset_cfg=SceneEntityCfg("robot")):
        """Shaped version of the pelvis half of ``standing``: 1 when the root's up axis is vertical."""
        g = env.scene[asset_cfg.name].data.projected_gravity_b
        return torch.exp(-(torch.square(g[:, 0]) + torch.square(g[:, 1])) / std**2)

    # Drop supine (roll pi) or prone (roll 0 with pitch pi/2 -> face down) from 0.35 m.
    cfg.events["reset_base"] = EventTermCfg(
        func=envs_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.45, -0.35),
                "roll": (math.pi - 0.3, math.pi + 0.3),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {},
        },
    )
    cfg.events.pop("push_robot", None)
    # Rewards: tracking and gait terms out, height + upright + standing in.
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
        "pose",
    ):
        cfg.rewards.pop(name, None)
    shape = GETUP_REWARDS[GETUP_REWARD]
    cfg.rewards["height"] = RewardTermCfg(
        func=height_reward, weight=shape["height_w"], params={"target": STAND_Z, "std": shape["height_std"]}
    )
    cfg.rewards["standing"] = RewardTermCfg(
        func=standing, weight=shape["standing_w"], params={"target": STAND_Z - shape["standing_dz"]}
    )
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].weight = shape["upright_w"]
    if shape.get("pelvis_upright_w"):
        cfg.rewards["pelvis_upright"] = RewardTermCfg(
            func=pelvis_upright, weight=shape["pelvis_upright_w"], params={"std": shape["pelvis_upright_std"]}
        )
    # Falling is the start state, so it cannot be a termination.
    cfg.terminations.pop("fell_over", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    # Command stays zero: this policy only has to stand up.
    twist = cfg.commands["twist"]
    for attr in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        if hasattr(twist.ranges, attr):
            setattr(twist.ranges, attr, (0.0, 0.0))
    if hasattr(twist, "rel_standing_envs"):
        twist.rel_standing_envs = 1.0
    cfg.episode_length_s = 6.0 if not play else 10.0
    return cfg


def register_getup() -> str:
    from mjlab.tasks.registry import list_tasks, load_rl_cfg, register_mjlab_task

    task_id = "Strands-GetUp-Unitree-G1"
    if task_id not in list_tasks():
        rl_cfg = load_rl_cfg(TASKS["getup"])
        register_mjlab_task(
            task_id=task_id, env_cfg=getup_env_cfg(), play_env_cfg=getup_env_cfg(play=True), rl_cfg=rl_cfg
        )
    return task_id


def task_id(task: str) -> str:
    return register_getup() if task == "getup" else TASKS[task]


# ------------------------------------------------------------------ train


def train(task: str, num_envs: int, iterations: int, run_dir: Path, seed: int) -> None:
    import torch  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    tid = task_id(task)
    env_cfg = load_env_cfg(tid)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    agent_cfg = load_rl_cfg(tid)
    agent_cfg.max_iterations = iterations
    agent_cfg.seed = seed
    agent_cfg.logger = "tensorboard"
    agent_cfg.save_interval = 100
    run_dir.mkdir(parents=True, exist_ok=True)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(tid) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), log_dir=str(run_dir), device="cuda:0")
    t0 = time.monotonic()
    runner.learn(iterations, init_at_random_ep_len=True)
    (run_dir / "train_done.json").write_text(
        json.dumps(
            {"task": tid, "num_envs": num_envs, "iterations": iterations, "wall_s": round(time.monotonic() - t0, 1)}
        ),
        encoding="utf-8",
    )
    env.close()


EXPORT_NCONMAX = 256  # contacts per world for the one-world export env (rough terrain overflows the heuristic)


def export(task: str, checkpoint: Path, onnx: Path, device: str = "cuda:0") -> None:
    """``strands_robots.training.mjlab_tasks.export.export_checkpoint`` with one change: an explicit nconmax.

    The core exporter builds the play env with ``num_envs = 1`` and leaves ``sim.nconmax`` on
    mjwarp's heuristic, which the rough-terrain G1 scene overflows at one world ("nconmax must
    be >= 72"). The dynamic-batch export and the metadata are the core's own helpers.
    """
    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    from strands_robots.training import mjlab_tasks
    from strands_robots.training.mjlab_tasks.export import _export_dynamic_batch

    mjlab_tasks.register_all()
    tid = task_id(task)
    env_cfg = load_env_cfg(tid, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.nconmax = EXPORT_NCONMAX
    agent_cfg = load_rl_cfg(tid)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(tid) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        _export_dynamic_batch(runner, Path(onnx))
        attach_metadata_to_onnx(str(onnx), get_base_metadata(env, f"g1_{task}"))
    finally:
        env.close()
    print(onnx)


# ------------------------------------------------------------ native eval


def eval_native(task: str, onnx: str, out: Path, ticks: int, seed: int, terrain: str = "generator") -> dict:
    """One mjlab world per command, play cfg (full terrain difficulty), ONNX actor via onnxruntime.

    ``terrain="plane"`` swaps the rough generator for a flat plane while keeping the ray-cast
    sensor, so the rough actor sees a real (flat) height scan: the control that separates
    "the actor is brittle" from "the classic-replay height_scan builder is wrong".
    """
    import onnxruntime as ort
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    tid = task_id(task)
    cfg = load_env_cfg(tid, play=True)
    commands = list(COMMANDS.items()) if task == "rough" else [("getup_supine", (0.0, 0.0, 0.0))] * 4
    cfg.scene.num_envs = len(commands)
    cfg.seed = seed
    if task == "rough" and cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        if terrain == "plane":
            cfg.scene.terrain.terrain_type = "plane"
            cfg.scene.terrain.terrain_generator = None
            cfg.sim.nconmax = EXPORT_NCONMAX
        else:
            cfg.scene.terrain.terrain_generator.difficulty_range = (0.9, 1.0)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    obs, _ = env.reset()
    robot = env.scene["robot"]
    cmd_t = torch.tensor([c for _, c in commands], device=env.device, dtype=torch.float32)
    twist = env.command_manager.get_term("twist")
    alive = torch.ones(len(commands), dtype=torch.bool, device=env.device)
    survived = torch.zeros(len(commands), device=env.device)
    v_err = torch.zeros(len(commands), device=env.device)
    z_sum = torch.zeros(len(commands), device=env.device)
    stood = torch.zeros(len(commands), device=env.device)
    z_max = torch.zeros(len(commands), device=env.device)
    upright_final = torch.zeros(len(commands), dtype=torch.bool, device=env.device)
    hz_int = int(round(1.0 / (env.cfg.sim.mujoco.timestep * env.cfg.decimation)))
    z_trace: list[list[float]] = [[] for _ in commands]
    for tick in range(ticks):
        twist.vel_command_b[:] = cmd_t
        act = sess.run(None, {in_name: obs["actor"].cpu().numpy().astype(np.float32)})[0]
        obs, _, terminated, time_out, _ = env.step(torch.as_tensor(act, device=env.device))
        alive &= ~(terminated.bool() | time_out.bool()) if task == "rough" else torch.ones_like(alive)
        z = robot.data.root_link_pos_w[:, 2]
        v_b = robot.data.root_link_lin_vel_b[:, :2]
        survived += alive.float()
        v_err += alive.float() * torch.linalg.norm(v_b - cmd_t[:, :2], dim=1)
        z_sum += alive.float() * z
        stood = torch.maximum(
            stood,
            ((z > STAND_Z - 0.1) & (robot.data.projected_gravity_b[:, 2] < -math.cos(math.radians(20.0)))).float(),
        )
        z_max = torch.maximum(z_max, z)
        upright_final = robot.data.projected_gravity_b[:, 2] < -math.cos(math.radians(20.0))
        if (tick + 1) % hz_int == 0:  # one pelvis height per second, so a trace reads as a story
            for i in range(len(commands)):
                z_trace[i].append(round(float(z[i]), 3))
    env.close()
    hz = 1.0 / (env.cfg.sim.mujoco.timestep * env.cfg.decimation)
    eps = {}
    for i, (name, cmd) in enumerate(commands):
        n = max(1.0, float(survived[i]))
        eps[f"{name}_{i}" if task != "rough" else name] = {
            "command": cmd,
            "survived_s": round(float(survived[i]) / hz, 2),
            "fell": bool(survived[i] < ticks),
            "v_xy_err_mean": round(float(v_err[i]) / n, 4),
            "base_z_mean": round(float(z_sum[i]) / n, 4),
            "base_z_max": round(float(z_max[i]), 4),
            "upright_final": bool(upright_final[i]),
            "z_per_second": z_trace[i],
            "stood_up": bool(stood[i] > 0),
        }
    rec = {
        "task": tid,
        "onnx": onnx,
        "terrain": terrain,
        "ticks": ticks,
        "hz": hz,
        "episodes": eps,
        "survived_all": all(not e["fell"] for e in eps.values()),
        "stood_up": sum(e["stood_up"] for e in eps.values()),
    }
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# --------------------------------------------------------------- s2s eval


_KNOWN_DIMS = {"command": 3, "base_ang_vel": 3, "base_lin_vel": 3, "projected_gravity": 3}


def _with_flat_height_scan(policy):
    """Teach the rsl_rl_onnx provider a ``height_scan`` term for a flat plane.

    Every ray hits z = 0, so each height is the pelvis height; the exporter's
    per-term scale (1 / max_distance) is applied by the provider itself. The ray
    count is what the actor's obs_dim leaves after the terms the provider knows
    (FINDINGS: the provider wants a pluggable term registry).
    """
    spec = policy.spec
    nj = len(spec.joint_names)
    known = sum(_KNOWN_DIMS.get(n, nj) for n in spec.observation_names if n != "height_scan")
    n_rays = spec.obs_dim - known
    original = policy._term

    def term(name, obs, kwargs):
        if name == "height_scan":
            z = float(np.asarray(obs.get("base_pos", [0, 0, STAND_Z]))[2])
            return np.full(n_rays, z, dtype=np.float32)
        return original(name, obs, kwargs)

    policy._term = term
    return policy, n_rays


async def eval_s2s(onnx: str, out: Path, ticks: int) -> dict:
    from strands_robots import Robot
    from strands_robots.policies import create_policy

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sim2sim_g1_velocity import rollout  # the deep lane's harness

    import strands_robots.policies.rsl_rl_onnx.policy as provider_module

    sim = Robot(ROBOT, backend="mujoco")
    # The provider refuses actors with observation terms it has no builder for, before any
    # instance can be patched; register the name first, then supply the flat-plane builder.
    if "height_scan" not in provider_module._BUILDERS:
        provider_module._BUILDERS = (*provider_module._BUILDERS, "height_scan")
    policy = create_policy("rsl_rl_onnx", onnx_path=onnx, robot=ROBOT)
    n_rays = 0
    if "height_scan" in policy.spec.observation_names:
        policy, n_rays = _with_flat_height_scan(policy)
    hz = 50.0
    eps = {}
    for name, cmd in COMMANDS.items():
        sim.reset()
        policy.reset()
        eps[name] = await rollout(sim, policy, cmd, ticks, hz)
    sim.cleanup()
    rec = {
        "onnx": onnx,
        "backend": "mujoco",
        "ticks": ticks,
        "height_scan_rays": n_rays,
        "episodes": eps,
        "survived_all": all(not e["fell"] for e in eps.values()),
    }
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# ------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--task", choices=TASKS, required=True)
    t.add_argument("--num-envs", type=int, default=2048)
    t.add_argument("--iterations", type=int, default=1500)
    t.add_argument("--run-dir", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--getup-reward", choices=GETUP_REWARDS, default=GETUP_REWARD, help="reward shape for --task getup")
    e = sub.add_parser("export")
    e.add_argument("--task", choices=TASKS, required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--onnx", required=True)
    n = sub.add_parser("eval-native")
    n.add_argument("--task", choices=TASKS, required=True)
    n.add_argument("--onnx", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--ticks", type=int, default=500)
    n.add_argument("--seed", type=int, default=7)
    n.add_argument("--terrain", choices=("generator", "plane"), default="generator")
    s = sub.add_parser("eval-s2s")
    s.add_argument("--onnx", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--ticks", type=int, default=500)
    a = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    if a.cmd == "train":
        globals()["GETUP_REWARD"] = a.getup_reward
        train(a.task, a.num_envs, a.iterations, Path(a.run_dir), a.seed)
    elif a.cmd == "export":
        export(a.task, Path(a.checkpoint), Path(a.onnx))
    elif a.cmd == "eval-native":
        rec = eval_native(a.task, a.onnx, Path(a.out), a.ticks, a.seed, a.terrain)
        print(
            json.dumps({k: v for k, v in rec.items() if k != "episodes"}),
            *(f"{k}: {v}" for k, v in rec["episodes"].items()),
            sep="\n",
        )
    else:
        rec = asyncio.run(eval_s2s(a.onnx, Path(a.out), a.ticks))
        print(
            json.dumps({k: v for k, v in rec.items() if k != "episodes"}),
            *(f"{k}: {v}" for k, v in rec["episodes"].items()),
            sep="\n",
        )


if __name__ == "__main__":
    main()
