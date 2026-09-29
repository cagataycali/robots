"""Every arm, one night: the same reach reward on every registry arm mjlab can build.

For each arm the script (a) inspects the MJCF (actuated joints, end-effector
site or leaf body), (b) samples the reachable set by MuJoCo forward kinematics
of random joint configurations, (c) builds the so101 reach task's reward stack
on that robot with targets drawn from the sampled set, (d) trains rsl_rl PPO
for a fixed budget, (e) exports ONNX with mjlab metadata, and (f) plays the
actor on the classic MuJoCo backend for N seeded targets (sim-to-sim). One
JSON per robot; ``--table`` folds them into the README table.

Install (the lerobot extra first, then this one: mjlab needs torch>=2.14)::

    uv pip install "strands-robots[sim-mjlab,rl]" matplotlib

Usage::

    python examples/mjlab/02_every_arm_one_night.py --robot ur5e --iterations 150 --num-envs 1024 --out runs/arms
    python examples/mjlab/02_every_arm_one_night.py --table runs/arms > table.md

Robots that fail to build or train are results too: their record carries an
``error`` and the stage that failed. Differences to the so101 task are
deliberate and minimal: joint terms are restricted to the actuated joints
(the ONNX provider rebuilds the observation from those), and the target
distribution is the measured reachable set instead of a hand-tuned box.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SUCCESS_M = 0.03
ACTION_SCALE = 0.25
EE_PATTERN = re.compile(r"(gripper|tcp|ee|end_effector|attachment|tool|flange|pinch|hand|grasp)", re.I)
METRIC = "Metrics/reach/at_goal"


# ------------------------------------------------------------- inspection


@dataclass
class ArmInfo:
    robot: str
    path: str
    joints: list[str]
    actuators: list[str]
    actuated_joints: list[str]
    ee_kind: str  # "site" or "body"
    ee_name: str
    free_base: bool
    notes: list[str] = field(default_factory=list)


def inspect_arm(robot: str) -> ArmInfo:
    """Read the registry MJCF: actuated joints (one hinge/slide per actuator) and an end effector."""
    import mujoco

    from strands_robots.assets import resolve_model_path, resolve_robot_name

    path = str(resolve_model_path(resolve_robot_name(robot)))
    m = mujoco.MjModel.from_xml_path(path)
    joints = [m.joint(j).name for j in range(m.njnt)]
    actuators = [m.actuator(a).name for a in range(m.nu)]
    notes: list[str] = []
    actuated: list[str] = []
    for a in range(m.nu):
        if int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            notes.append(f"actuator {actuators[a]} is not a joint transmission (trntype {int(m.actuator_trntype[a])})")
            continue
        j = int(m.actuator_trnid[a, 0])
        if int(m.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            notes.append(f"actuator {actuators[a]} drives a non scalar joint")
            continue
        actuated.append(m.joint(j).name)
    free_base = m.njnt > 0 and int(m.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
    sites = [m.site(s).name for s in range(m.nsite)]
    ee_kind, ee_name = "site", ""
    for s in sites:
        if EE_PATTERN.search(s):
            ee_name = s
            break
    if not ee_name:
        # Leaf body furthest down the kinematic tree, preferring an end-effector-like name.
        children = {b: 0 for b in range(m.nbody)}
        for b in range(1, m.nbody):
            children[int(m.body_parentid[b])] += 1
        leaves = [b for b in range(1, m.nbody) if children[b] == 0]
        depth = {}
        for b in range(m.nbody):
            d, p = 0, b
            while p != 0:
                p = int(m.body_parentid[p])
                d += 1
            depth[b] = d
        named = [b for b in leaves if EE_PATTERN.search(m.body(b).name)]
        pick = max(named or leaves, key=lambda b: depth[b])
        ee_kind, ee_name = "body", m.body(pick).name
        notes.append(f"no end-effector site; using leaf body {ee_name!r}")
    return ArmInfo(robot, path, joints, actuators, actuated, ee_kind, ee_name, free_base, notes)


class ArmFK:
    """Forward kinematics of the end effector in the robot base frame (classic MuJoCo, CPU)."""

    def __init__(self, info: ArmInfo) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(info.path)
        self.data = mujoco.MjData(self.model)
        self.qadr = [
            int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)])
            for j in info.actuated_joints
        ]
        obj = mujoco.mjtObj.mjOBJ_SITE if info.ee_kind == "site" else mujoco.mjtObj.mjOBJ_BODY
        self.ee_id = mujoco.mj_name2id(self.model, obj, info.ee_name)
        self.kind = info.ee_kind
        # The base frame is the first body under the world.
        self.base_id = 1 if self.model.nbody > 1 else 0

    def site_pos(self, q: list[float]) -> np.ndarray:
        """End-effector position in the base frame for actuated-joint positions ``q`` (provider hook name)."""
        self.data.qpos[:] = 0.0
        for adr, v in zip(self.qadr, q, strict=True):
            self.data.qpos[adr] = v
        self.mujoco.mj_kinematics(self.model, self.data)
        ee = self.data.site_xpos[self.ee_id] if self.kind == "site" else self.data.xpos[self.ee_id]
        base_p = self.data.xpos[self.base_id]
        base_r = self.data.xmat[self.base_id].reshape(3, 3)
        return (base_r.T @ (ee - base_p)).astype(np.float32)

    def sample_reachable(self, n: int, seed: int, z_min: float = 0.03) -> np.ndarray:
        """FK of ``n`` uniformly random joint configurations, above the floor, base frame."""
        rng = np.random.default_rng(seed)
        lo, hi = [], []
        for j in self.qadr:
            jid = int(np.where(self.model.jnt_qposadr == j)[0][0])
            if self.model.jnt_limited[jid]:
                lo.append(self.model.jnt_range[jid, 0])
                hi.append(self.model.jnt_range[jid, 1])
            else:
                lo.append(-np.pi)
                hi.append(np.pi)
        pts = []
        base_z = float(self.model.body_pos[self.base_id, 2])
        for _ in range(n):
            q = rng.uniform(lo, hi)
            p = self.site_pos(list(q))
            if p[2] + base_z > z_min:
                pts.append(p)
        return np.asarray(pts, dtype=np.float32)


# ------------------------------------------------------------- mjlab task


def build_task(info: ArmInfo, cloud: np.ndarray, *, play: bool = False):
    """The so101 reach task on ``info``'s robot, targets drawn from ``cloud``."""
    import torch
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.tasks.velocity import mdp
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.utils.noise import UniformNoiseCfg as Unoise
    from mjlab.viewer import ViewerConfig

    from strands_robots.simulation.mjlab.simulation import MjlabEngine, _RobotSpec
    from strands_robots.training.mjlab_tasks import so101_reach as base

    cloud_t = torch.as_tensor(cloud)

    class CloudReachCommand(base.ReachCommand):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.cloud = cloud_t.to(self.device)

        def ee_pos_w(self) -> torch.Tensor:
            if info.ee_kind == "site":
                if not hasattr(self, "_ee_site_id"):
                    self._ee_site_id = self.robot.find_sites(f"^{re.escape(info.ee_name)}$")[0][0]
                return self.robot.data.site_pos_w[:, self._ee_site_id]
            if not hasattr(self, "_ee_body_id"):
                self._ee_body_id = self.robot.find_bodies(f"^{re.escape(info.ee_name)}$")[0][0]
            return self.robot.data.body_link_pos_w[:, self._ee_body_id]

        def _resample_command(self, env_ids: torch.Tensor) -> None:
            idx = torch.randint(0, len(self.cloud), (len(env_ids),), device=self.device)
            self.target_pos_b[env_ids] = self.cloud[idx]

    @dataclass(kw_only=True)
    class CloudReachCommandCfg(base.ReachCommandCfg):
        def build(self, env):
            return CloudReachCommand(self, env)

    spec = _RobotSpec(
        name="robot",
        path=info.path,
        position=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        keyframe=None,
        actuator_names=list(info.actuators),
    )
    entity = MjlabEngine._robot_entity_cfg(None, spec)  # type: ignore[arg-type]
    # Natural (MJCF) joint order, the order the ONNX metadata and the provider use.
    act_cfg = SceneEntityCfg("robot", joint_names=tuple(f"^{re.escape(j)}$" for j in info.actuated_joints))
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, params={"asset_cfg": act_cfg}, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, params={"asset_cfg": act_cfg}, noise=Unoise(n_min=-0.5, n_max=0.5)
        ),
        "ee_to_target": ObservationTermCfg(func=base.ee_to_target, noise=Unoise(n_min=-0.005, n_max=0.005)),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg({**actor_terms}, enable_corruption=not play),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }
    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",), scale=ACTION_SCALE, use_default_offset=True
        )
    }
    commands = {"reach": CloudReachCommandCfg(resampling_time_range=(3.0, 5.0), debug_vis=play)}
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {}, "velocity_range": {}}
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (-0.3, 0.3), "velocity_range": (0.0, 0.0), "asset_cfg": act_cfg},
        ),
    }
    rewards = {
        "reach_coarse": RewardTermCfg(func=base.reach_reward, weight=1.0, params={"std": 0.15}),
        "reach_fine": RewardTermCfg(func=base.reach_reward, weight=2.0, params={"std": 0.03}),
        "success": RewardTermCfg(func=base.reach_success, weight=1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": act_cfg}),
    }
    terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1, env_spacing=2.5, entities={"robot": entity}
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.WORLD, distance=2.0),
        sim=SimulationCfg(
            nconmax=100, njmax=800, mujoco=MujocoCfg(timestep=0.002, iterations=10, ls_iterations=20, cone="elliptic")
        ),
        decimation=10,
        episode_length_s=8.0 if not play else 20.0,
    )


# ----------------------------------------------------------------- train


def train_and_export(info: ArmInfo, cloud: np.ndarray, *, num_envs: int, iterations: int, seed: int, out: Path) -> dict:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
    from mjlab.rl.runner import MjlabOnPolicyRunner

    from strands_robots.training.mjlab_tasks.export import _export_dynamic_batch
    from strands_robots.training.mjlab_tasks.so101_reach import so101_reach_ppo_runner_cfg

    env_cfg = build_task(info, cloud)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    agent_cfg = so101_reach_ppo_runner_cfg()
    agent_cfg.max_iterations = iterations
    agent_cfg.save_interval = iterations
    agent_cfg.seed = seed
    agent_cfg.logger = "tensorboard"
    agent_cfg.experiment_name = f"arm_reach_{info.robot}"
    log_dir = out / info.robot / "train"
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), log_dir=str(log_dir), device="cuda:0")
    build_s = time.monotonic() - t0
    history: list[dict] = []
    logger = runner.logger
    original_log = logger.log

    def hooked_log(*, it: int, collect_time: float, learn_time: float, **kw) -> None:
        vals = [float(torch.as_tensor(e[METRIC]).float().mean()) for e in logger.ep_extras if METRIC in e]
        history.append(
            {
                "it": it,
                "at_goal": round(sum(vals) / len(vals), 4) if vals else None,
                "fps": int(agent_cfg.num_steps_per_env * num_envs / (collect_time + learn_time)),
            }
        )
        original_log(it=it, collect_time=collect_time, learn_time=learn_time, **kw)

    logger.log = hooked_log  # type: ignore[method-assign]
    t1 = time.monotonic()
    runner.learn(iterations, init_at_random_ep_len=True)
    train_s = time.monotonic() - t1
    onnx_path = out / info.robot / f"{info.robot}_reach.onnx"
    _export_dynamic_batch(runner, onnx_path)
    md = get_base_metadata(env, f"arm_reach_{info.robot}")
    # mjlab lists every joint of the entity; the actor only drives the actuated ones and
    # the provider refuses a joint_names / outputs mismatch (FINDINGS F3). Filter in
    # mjlab's natural joint order, which is the order the joint observation terms use.
    keep = [i for i, j in enumerate(md["joint_names"]) if j in info.actuated_joints]
    md["joint_names"] = [md["joint_names"][i] for i in keep]
    md["default_joint_pos"] = [md["default_joint_pos"][i] for i in keep]
    attach_metadata_to_onnx(str(onnx_path), md)
    obs_dim = int(sum(env.observation_manager.group_obs_dim["actor"]))
    env.close()
    tail = [h["at_goal"] for h in history[-10:] if h["at_goal"] is not None]
    return {
        "build_s": round(build_s, 1),
        "train_s": round(train_s, 1),
        "iterations": len(history),
        "fps_median": int(np.median([h["fps"] for h in history])) if history else 0,
        "at_goal_last10": round(float(np.mean(tail)), 4) if tail else None,
        "at_goal_max": max((h["at_goal"] for h in history if h["at_goal"] is not None), default=None),
        "obs_dim": obs_dim,
        "onnx": str(onnx_path),
        "history": history,
    }


# --------------------------------------------------------------- sim2sim


async def sim2sim(info: ArmInfo, onnx: str, targets: np.ndarray, *, ticks: int = 150, hz: float = 50.0) -> dict:
    """Play the ONNX actor on the classic MuJoCo backend for each target; success = ee within SUCCESS_M."""
    import strands_robots.policies.rsl_rl_onnx.policy as provider_module
    from strands_robots import Robot
    from strands_robots.policies import create_policy

    fk = ArmFK(info)
    sim = Robot(info.robot, backend="mujoco")
    # The provider only knows MJCF sites and reads them from the registry MJCF; the
    # example's FK also handles a leaf body and the base frame, so it is injected
    # (FINDINGS: the provider wants an injectable FK / body fallback).
    original_fk = provider_module._SiteFK
    provider_module._SiteFK = lambda robot, site, joints: fk  # type: ignore[assignment]
    try:
        policy = create_policy("rsl_rl_onnx", onnx_path=onnx, robot=info.robot)
    finally:
        provider_module._SiteFK = original_fk
    joints = list(policy.spec.joint_names)
    n_sub = max(1, int(round(sim.physics_timestep() ** -1 / hz)))
    eps = []
    for target in targets:
        sim.reset()
        policy.reset()
        errs = []
        for _ in range(ticks):
            obs = sim.get_observation(info.robot)
            q = [float(obs[j]) for j in joints]
            errs.append(float(np.linalg.norm(fk.site_pos(q) - target)))
            chunk = await policy.get_actions(obs, "", target_pose=target.tolist())
            sim.send_action(chunk[0], info.robot, n_substeps=n_sub)
        obs = sim.get_observation(info.robot)
        final = float(np.linalg.norm(fk.site_pos([float(obs[j]) for j in joints]) - target))
        eps.append({"final_err_m": final, "min_err_m": float(min(errs + [final])), "success": final < SUCCESS_M})
    sim.cleanup()
    succ = sum(e["success"] for e in eps)
    return {
        "backend": "mujoco",
        "n": len(eps),
        "success": f"{succ}/{len(eps)}",
        "success_rate": succ / len(eps),
        "final_err_median_m": float(np.median([e["final_err_m"] for e in eps])),
        "episodes": eps,
    }


# ------------------------------------------------------------------ main


def run_robot(robot: str, *, num_envs: int, iterations: int, seed: int, out: Path, n_eval: int) -> dict:
    rec: dict = {"robot": robot, "stage": "inspect", "t_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / robot).mkdir(parents=True, exist_ok=True)
    try:
        info = inspect_arm(robot)
        rec.update(
            {
                "dof": len(info.actuated_joints),
                "actuators": len(info.actuators),
                "ee": f"{info.ee_kind}:{info.ee_name}",
                "notes": info.notes,
            }
        )
        if not info.actuated_joints:
            raise RuntimeError("no scalar-joint actuators in the MJCF")
        if info.free_base:
            raise RuntimeError("floating base; not an arm reach setup")
        rec["stage"] = "reachable_set"
        fk = ArmFK(info)
        cloud = fk.sample_reachable(4000, seed)
        if len(cloud) < 200:
            raise RuntimeError(f"reachable set too small ({len(cloud)} of 4000 samples above the floor)")
        rec["reachable"] = {
            "n": int(len(cloud)),
            "min": np.round(cloud.min(0), 3).tolist(),
            "max": np.round(cloud.max(0), 3).tolist(),
            "reach_m": round(float(np.linalg.norm(cloud, axis=1).max()), 3),
        }
        rec["stage"] = "train"
        rec["train"] = train_and_export(info, cloud, num_envs=num_envs, iterations=iterations, seed=seed, out=out)
        rec["stage"] = "sim2sim"
        eval_targets = cloud[np.random.default_rng(seed + 1).choice(len(cloud), n_eval, replace=False)]
        rec["sim2sim"] = asyncio.run(sim2sim(info, rec["train"]["onnx"], eval_targets))
        rec["stage"] = "done"
    except Exception as exc:  # a failed robot is a finding, not a crash of the night
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    rec["t_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / robot / "result.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def table(out: Path) -> str:
    rows = []
    for f in sorted(out.glob("*/result.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        tr, s2 = r.get("train") or {}, r.get("sim2sim") or {}
        rows.append(
            f"| {r['robot']} | {r.get('dof', '')} | {r.get('ee', '')} | "
            f"{round(tr['train_s'] / 60, 1) if tr else ''} | {tr.get('fps_median', '') if tr else ''} | "
            f"{tr.get('at_goal_last10', '') if tr else ''} | {s2.get('success', '')} | "
            f"{round(s2['final_err_median_m'] * 1000) if s2 else ''} | {r.get('error', '') if 'error' in r else 'ok'} |"
        )
    head = "| robot | DoF | end effector | train min | env-steps/s | at_goal (last 10 its) | s2s mujoco | median err mm | status |\n|---|---|---|---|---|---|---|---|---|"
    return head + "\n" + "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", nargs="*", default=[])
    p.add_argument("--all-arms", action="store_true", help="every registry arm with has_sim")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-eval", type=int, default=20)
    p.add_argument("--out", default="runs/arms")
    p.add_argument("--table", help="fold <out>/*/result.json into a markdown table")
    a = p.parse_args(argv)
    if a.table:
        print(table(Path(a.table)))
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    robots = list(a.robot)
    if a.all_arms:
        from strands_robots.assets import list_robots

        robots += [
            r["name"] for r in list_robots() if r["category"] == "arm" and r["has_sim"] and r["name"] not in robots
        ]
    for robot in robots:
        rec = run_robot(
            robot, num_envs=a.num_envs, iterations=a.iterations, seed=a.seed, out=Path(a.out), n_eval=a.n_eval
        )
        summary = {k: v for k, v in rec.items() if k not in ("train", "sim2sim")}
        if "train" in rec:
            summary["train"] = {k: v for k, v in rec["train"].items() if k != "history"}
        if "sim2sim" in rec:
            summary["sim2sim"] = {k: v for k, v in rec["sim2sim"].items() if k != "episodes"}
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
