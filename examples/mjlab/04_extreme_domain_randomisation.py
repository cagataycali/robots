"""Domain randomisation to the extreme: per-world physics that no single-model simulator can offer.

Classic MuJoCo has one model, so randomising friction or gains means recompiling
or mutating it for every episode of every environment. mjlab (MuJoCo Warp) keeps
a model *per world*, so every one of N worlds can carry its own friction, mass,
PD gains, effort limits, encoder bias and joint damping, redrawn on every reset.

This example trains the so101 reach task twice through the mjlab trainer, once
with the task's defaults and once with an extreme randomisation set (see
``EXTREME``), exports both actors to ONNX, then evaluates both on the classic
MuJoCo backend across a grid of physical perturbations the policies never saw
(payload on the gripper, mass scale, friction, PD gain scale, joint damping).
The question is not "does DR help" in general, it is "how far can it go before
the nominal policy breaks and the DR policy does not"; the JSON holds the answer.

Stages::

    python examples/mjlab/04_extreme_domain_randomisation.py train --dr none --num-envs 1024 --iterations 300 --run-dir runs/reach_nominal
    python examples/mjlab/04_extreme_domain_randomisation.py train --dr extreme --num-envs 1024 --iterations 300 --run-dir runs/reach_dr
    python examples/mjlab/04_extreme_domain_randomisation.py export --dr extreme --checkpoint runs/reach_dr/model_299.pt --onnx runs/reach_dr.onnx
    python examples/mjlab/04_extreme_domain_randomisation.py eval --nominal runs/reach_nominal.onnx --dr runs/reach_dr.onnx --out dr_eval.json

Install (the lerobot extra first, then this one: mjlab needs torch>=2.14)::

    uv pip install "strands-robots[sim-mjlab,rl]"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROBOT = "so101"
JOINTS = ["1", "2", "3", "4", "5", "6"]
TASK_IDS = {"none": "Strands-Reach-So101", "extreme": "Strands-Reach-So101-DR-Extreme"}

# The extreme set. Every entry is redrawn per world on every reset (mode="reset").
EXTREME = {
    "pd_gains": {"kp": (0.4, 1.8), "kd": (0.4, 1.8)},
    "effort_limits": (0.4, 1.0),
    "body_mass": (0.5, 2.0),
    "geom_friction": (0.1, 2.0),
    "joint_damping": (0.3, 4.0),
    "encoder_bias_rad": (-0.05, 0.05),
    "obs_noise_scale": 3.0,
}

# The classic-MuJoCo perturbation grid the policies are judged on (never seen in training).
PERTURBATIONS = {
    "nominal": {},
    "payload_100g": {"payload_kg": 0.1},
    "payload_250g": {"payload_kg": 0.25},
    "mass_x2": {"mass_scale": 2.0},
    "friction_0.1": {"friction": 0.1},
    "kp_x0.5": {"kp_scale": 0.5},
    "kp_x1.8": {"kp_scale": 1.8},
    "damping_x4": {"damping_scale": 4.0},
    "everything": {"payload_kg": 0.1, "mass_scale": 1.5, "friction": 0.3, "kp_scale": 0.6, "damping_scale": 3.0},
    # Second wave, added after the first nine cells came back flat (F10): a
    # position-observed actor integrates any static sag away, so these cells
    # attack the loop itself - what the observation says and when the action lands.
    "encoder_bias_50mrad": {"encoder_bias_rad": 0.05},
    "encoder_bias_100mrad": {"encoder_bias_rad": 0.10},
    "obs_noise_20mrad": {"obs_noise_rad": 0.02},
    "action_delay_2": {"action_delay_ticks": 2},
    "action_delay_5": {"action_delay_ticks": 5},
    "hostile": {
        "payload_kg": 0.1,
        "kp_scale": 0.6,
        "encoder_bias_rad": 0.05,
        "obs_noise_rad": 0.02,
        "action_delay_ticks": 2,
    },
}
# Keys of PERTURBATIONS that live in the observation/action loop, not the MjModel.
LOOP_KEYS = ("encoder_bias_rad", "obs_noise_rad", "action_delay_ticks")


# ------------------------------------------------------------------- tasks


def extreme_env_cfg(play: bool = False):
    """so101 reach cfg with the EXTREME randomisation set attached (mode=reset, per world)."""
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from strands_robots.training.mjlab_tasks.so101_reach import so101_reach_env_cfg

    cfg = so101_reach_env_cfg(play=play)
    robot = SceneEntityCfg("robot")
    cfg.events["dr_pd_gains"] = EventTermCfg(
        func=dr.pd_gains,
        mode="reset",
        params={
            "asset_cfg": robot,
            "kp_range": EXTREME["pd_gains"]["kp"],
            "kd_range": EXTREME["pd_gains"]["kd"],
            "operation": "scale",
        },
    )
    cfg.events["dr_effort"] = EventTermCfg(
        func=dr.effort_limits,
        mode="reset",
        params={"asset_cfg": robot, "effort_limit_range": EXTREME["effort_limits"], "operation": "scale"},
    )
    cfg.events["dr_mass"] = EventTermCfg(
        func=dr.body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
            "operation": "scale",
            "ranges": EXTREME["body_mass"],
        },
    )
    cfg.events["dr_friction"] = EventTermCfg(
        func=dr.geom_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
            "operation": "abs",
            "ranges": EXTREME["geom_friction"],
        },
    )
    cfg.events["dr_damping"] = EventTermCfg(
        func=dr.joint_damping,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "scale",
            "ranges": EXTREME["joint_damping"],
        },
    )
    cfg.events["dr_encoder_bias"] = EventTermCfg(
        func=dr.encoder_bias, mode="reset", params={"asset_cfg": robot, "bias_range": EXTREME["encoder_bias_rad"]}
    )
    if not play:
        for term in cfg.observations["actor"].terms.values():
            noise = getattr(term, "noise", None)
            if noise is not None and hasattr(noise, "n_min"):
                noise.n_min *= EXTREME["obs_noise_scale"]
                noise.n_max *= EXTREME["obs_noise_scale"]
    return cfg


def task_id(dr: str) -> str:
    from mjlab.tasks.registry import list_tasks, load_rl_cfg, register_mjlab_task

    from strands_robots.training.mjlab_tasks.so101_reach import register

    base = register()
    if dr == "none":
        return base
    tid = TASK_IDS["extreme"]
    if tid not in list_tasks():
        register_mjlab_task(
            task_id=tid, env_cfg=extreme_env_cfg(), play_env_cfg=extreme_env_cfg(play=True), rl_cfg=load_rl_cfg(base)
        )
    return tid


# ------------------------------------------------------------------- train


def train(dr: str, num_envs: int, iterations: int, run_dir: Path, seed: int) -> None:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    tid = task_id(dr)
    env_cfg = load_env_cfg(tid)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    agent_cfg = load_rl_cfg(tid)
    agent_cfg.max_iterations = iterations
    agent_cfg.seed = seed
    agent_cfg.logger = "tensorboard"
    agent_cfg.save_interval = 50
    run_dir.mkdir(parents=True, exist_ok=True)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), log_dir=str(run_dir), device="cuda:0")
    t0 = time.monotonic()
    runner.learn(iterations, init_at_random_ep_len=True)
    (run_dir / "train_done.json").write_text(
        json.dumps(
            {
                "task": tid,
                "dr": dr,
                "num_envs": num_envs,
                "iterations": iterations,
                "wall_s": round(time.monotonic() - t0, 1),
                "extreme": EXTREME if dr == "extreme" else None,
            }
        ),
        encoding="utf-8",
    )
    env.close()


def export(dr: str, checkpoint: Path, onnx: Path) -> None:
    from strands_robots.training.mjlab_tasks.export import export_checkpoint

    print(export_checkpoint(task_id(dr), checkpoint, onnx, run_name=f"reach_{dr}"))


# -------------------------------------------------------------------- eval


def perturb(sim, spec: dict) -> dict:
    """Mutate the classic MuJoCo model in place (between steps, single-threaded here) and report what changed."""
    import mujoco

    m = sim.mj_model
    applied = {}
    if "payload_kg" in spec:
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        if body < 0:
            body = m.nbody - 1
        m.body_mass[body] += spec["payload_kg"]
        m.body_inertia[body] *= (m.body_mass[body]) / max(m.body_mass[body] - spec["payload_kg"], 1e-6)
        applied["payload_body"] = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body)
    if "mass_scale" in spec:
        m.body_mass[1:] *= spec["mass_scale"]
        m.body_inertia[1:] *= spec["mass_scale"]
    if "friction" in spec:
        m.geom_friction[:, 0] = spec["friction"]
    if "kp_scale" in spec:
        m.actuator_gainprm[:, 0] *= spec["kp_scale"]
        m.actuator_biasprm[:, 1] *= spec["kp_scale"]
    if "damping_scale" in spec:
        m.dof_damping[:] *= spec["damping_scale"]
    mujoco.mj_setConst(m, sim.mj_data)
    applied.update(spec)
    return applied


async def rollout(sim, policy, fk, target, ticks: int, hz: float, spec: dict) -> dict:
    from strands_robots.training.mjlab_tasks.so101_reach import SUCCESS_M

    sim.reset()
    perturb(sim, {k: v for k, v in spec.items() if k not in LOOP_KEYS})
    policy.reset()
    n_sub = max(1, int(round(sim.physics_timestep() ** -1 / hz)))
    bias = float(spec.get("encoder_bias_rad", 0.0))
    noise = float(spec.get("obs_noise_rad", 0.0))
    delay = int(spec.get("action_delay_ticks", 0))
    rng = np.random.default_rng(int(round(1000 * float(target[0]) + 7)))
    queue: list = []  # actions in flight when the loop has latency
    errs = []
    for _ in range(ticks):
        obs = sim.get_observation(ROBOT)
        q = [float(obs[j]) for j in JOINTS]
        errs.append(float(np.linalg.norm(fk.site_pos(q) - target)))
        if bias or noise:
            # The policy sees a corrupted encoder; the truth (errs above) stays uncorrupted.
            seen = dict(obs)
            for j in JOINTS:
                seen[j] = float(obs[j]) + bias + (float(rng.normal(0.0, noise)) if noise else 0.0)
        else:
            seen = obs
        chunk = await policy.get_actions(seen, "", target_pose=target.tolist())
        queue.append(chunk[0])
        if len(queue) > delay:
            sim.send_action(queue.pop(0), ROBOT, n_substeps=n_sub)
        else:
            sim.send_action(
                {j: float(obs[j]) for j in JOINTS}, ROBOT, n_substeps=n_sub
            )  # nothing has arrived yet: hold
    obs = sim.get_observation(ROBOT)
    q = [float(obs[j]) for j in JOINTS]
    final = float(np.linalg.norm(fk.site_pos(q) - target))
    return {
        "final_err_m": round(final, 4),
        "min_err_m": round(float(min(errs + [final])), 4),
        "success": final < SUCCESS_M,
    }


async def evaluate(
    onnx_by_name: dict[str, str], n: int, seed: int, ticks: int, hz: float, out: Path, only: tuple[str, ...] = ()
) -> dict:
    import sys

    from strands_robots import Robot
    from strands_robots.policies import create_policy
    from strands_robots.policies.rsl_rl_onnx.policy import _SiteFK
    from strands_robots.training.mjlab_tasks.so101_reach import SUCCESS_M

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sim2sim_reach import sample_targets

    targets = sample_targets(n, seed)
    fk = _SiteFK(ROBOT, "gripper", JOINTS)
    table: dict[str, dict[str, dict]] = {}
    if out.exists():
        # Resume / extend: cells already measured are kept, only missing ones run.
        table = json.loads(out.read_text(encoding="utf-8")).get("table", {})
    for pname, spec in PERTURBATIONS.items():
        if only and pname not in only:
            continue
        if pname in table and all(p in table[pname] for p in onnx_by_name):
            continue
        table[pname] = {}
        for label, onnx in onnx_by_name.items():
            policy = create_policy("rsl_rl_onnx", onnx_path=onnx, robot=ROBOT)
            eps = []
            for t in targets:
                sim = Robot(ROBOT, backend="mujoco")  # fresh model per episode so perturbations never stack
                eps.append(await rollout(sim, policy, fk, t, ticks, hz, spec))
                sim.cleanup()
            succ = sum(e["success"] for e in eps)
            table[pname][label] = {
                "success": f"{succ}/{len(eps)}",
                "success_rate": succ / len(eps),
                "final_err_median_m": round(float(np.median([e["final_err_m"] for e in eps])), 4),
                "episodes": eps,
            }
            print(
                f"{pname:14s} {label:8s} success {succ:>2d}/{len(eps)}  median final err {table[pname][label]['final_err_median_m']:.4f} m",
                flush=True,
            )
    rec = {
        "backend": "mujoco",
        "n": n,
        "seed": seed,
        "ticks": ticks,
        "hz": hz,
        "success_m": SUCCESS_M,
        "policies": onnx_by_name,
        "extreme": EXTREME,
        "perturbations": PERTURBATIONS,
        "table": table,
    }
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dr", choices=TASK_IDS, required=True)
    t.add_argument("--num-envs", type=int, default=1024)
    t.add_argument("--iterations", type=int, default=300)
    t.add_argument("--run-dir", required=True)
    t.add_argument("--seed", type=int, default=42)
    e = sub.add_parser("export")
    e.add_argument("--dr", choices=TASK_IDS, required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--onnx", required=True)
    v = sub.add_parser("eval")
    v.add_argument("--nominal", required=True, help="ONNX of the policy trained with --dr none")
    v.add_argument("--dr", required=True, help="ONNX of the policy trained with --dr extreme")
    v.add_argument("--out", required=True)
    v.add_argument("--n", type=int, default=20)
    v.add_argument("--seed", type=int, default=7)
    v.add_argument("--ticks", type=int, default=200)
    v.add_argument("--hz", type=float, default=50.0)
    v.add_argument(
        "--only", nargs="*", default=(), help="restrict to these perturbation names (cells already in --out are kept)"
    )
    a = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    if a.cmd == "train":
        train(a.dr, a.num_envs, a.iterations, Path(a.run_dir), a.seed)
    elif a.cmd == "export":
        export(a.dr, Path(a.checkpoint), Path(a.onnx))
    else:
        asyncio.run(
            evaluate({"nominal": a.nominal, "dr": a.dr}, a.n, a.seed, a.ticks, a.hz, Path(a.out), tuple(a.only))
        )


if __name__ == "__main__":
    main()
