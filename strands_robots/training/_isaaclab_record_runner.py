"""Roll a trained rsl_rl policy out in Isaac Lab and write each episode as raw arrays.

Run by the Isaac Lab interpreter (``ISAACLAB_PYTHON``), never imported by
strands-robots: it imports only Isaac Lab, rsl_rl, torch and numpy, so the
separate-venv design holds. :meth:`strands_robots.training.isaaclab.IsaacLabTrainer.record`
launches it, and the parent converts what it writes into a LeRobotDataset with
strands' own :class:`~strands_robots.dataset_recorder.DatasetRecorder`.

Environment ``i`` of ``--episodes`` parallel environments is episode ``i``: every
one starts from the same reset and ends at its first ``done`` or at
``--frames``. Written to ``--out``:

* ``episode_<i>.npz`` - ``joint_pos`` ``(T, J)``, ``policy_obs`` ``(T, D)``,
  ``action`` ``(T, A)``, ``root_pos`` / ``root_quat`` ``(T, 3/4)`` (position
  relative to the env origin, quaternion as Isaac Lab reports it, x-y-z-w),
  ``reward`` ``(T,)`` and, with a camera, ``image`` ``(T, H, W, 3)`` uint8;
* ``meta.json`` - joint and action names, fps, task, checkpoint, each
  episode's length, return and how it ended; written last, so its presence
  means the rollout finished.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# Run as a script from inside strands_robots/training/, whose isaaclab.py would
# otherwise shadow the real ``isaaclab`` package on ``sys.path[0]``.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]


def _look_quat_world(eye: list[float], target: list[float]) -> tuple[float, float, float, float]:
    """x-y-z-w quaternion, Isaac Lab's ``world`` camera convention (+X forward, +Z up), eye -> target."""
    dx, dy, dz = (t - e for t, e in zip(target, eye, strict=True))
    yaw, pitch = math.atan2(dy, dx), math.atan2(-dz, math.hypot(dx, dy))
    sz, cz, sy, cy = math.sin(yaw / 2), math.cos(yaw / 2), math.sin(pitch / 2), math.cos(pitch / 2)
    return (-sz * sy, cz * sy, sz * cy, cz * cy)


def _tensor(value):  # type: ignore[no-untyped-def]
    return value.torch if hasattr(value, "torch") else value


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--camera", choices=["fixed", "none"], default="fixed")
    ap.add_argument("--eye", default="2.5,-2.5,1.5", help="camera position, from each env origin")
    ap.add_argument("--target", default="0,0,0.3", help="camera look-at, from each env origin")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.argv = sys.argv[:1]  # Kit parses argv too

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401 - registers the tasks
    import numpy as np
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import resolve_task_config

    os.makedirs(a.out, exist_ok=True)
    n = a.episodes
    env_cfg, agent_cfg = resolve_task_config(a.task, a.agent, play_mode=True, overrides=list(a.override))
    env_cfg.scene.num_envs = n
    env_cfg.seed = a.seed
    use_cam = a.camera != "none"
    if use_cam:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg

        eye = [float(v) for v in a.eye.split(",")]
        target = [float(v) for v in a.target.split(",")]
        env_cfg.scene.strands_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/StrandsCam",
            update_period=0.0,
            height=a.height,
            width=a.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.02, 50.0)),
            offset=CameraCfg.OffsetCfg(pos=tuple(eye), rot=_look_quat_world(eye, target), convention="world"),
        )
    t0 = time.time()
    with launch_simulation(env_cfg, {"headless": True, "enable_cameras": use_cam}):
        import importlib.metadata as md

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, md.version("rsl-rl-lib"))
        genv = gym.make(a.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(genv, clip_actions=agent_cfg.clip_actions)
        u = env.unwrapped
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(u.device))
        runner.load(a.checkpoint)
        policy = runner.get_inference_policy(device=str(u.device))
        groups = list(getattr(runner.alg.get_policy(), "obs_groups", ["policy"]))
        robot = next(iter(u.scene.articulations.values()))
        joint_names = list(robot.joint_names)
        adim = int(env.num_actions)
        names: list[str] = []
        manager = getattr(u, "action_manager", None)
        if manager is not None:
            for term_name, term in manager._terms.items():
                joints = list(getattr(term, "_joint_names", []) or [])
                names += (
                    [f"{term_name}.{j}" for j in joints]
                    if len(joints) == term.action_dim
                    else [f"{term_name}.{k}" for k in range(term.action_dim)]
                )
        if len(names) != adim:
            names = [f"a{k:02d}" for k in range(adim)]
        fps = round(1.0 / u.step_dt)
        buf: dict[str, list[list]] = {k: [[] for _ in range(n)] for k in ("q", "o", "a", "p", "r4", "rw", "img")}
        alive = np.ones(n, bool)
        ret = np.zeros(n)
        lens = np.zeros(n, int)
        ended = ["frames"] * n
        with torch.inference_mode():
            env.step(torch.zeros(n, adim, device=u.device))  # warm the renderer
            env.reset()
        obs = env.get_observations()
        for _ in range(a.frames):
            with torch.inference_mode():
                actions = policy(obs)
            q = _tensor(robot.data.joint_pos).cpu().numpy()
            pobs = torch.cat([obs[g] for g in groups], dim=-1).cpu().numpy()
            rp = _tensor(robot.data.root_pos_w).cpu().numpy() - u.scene.env_origins.cpu().numpy()
            rq = _tensor(robot.data.root_quat_w).cpu().numpy()
            img = None
            if use_cam:
                img = _tensor(u.scene["strands_cam"].data.output["rgb"])[..., :3].to(torch.uint8).cpu().numpy()
            act = actions.cpu().numpy()
            with torch.inference_mode():
                obs, rew, dones, extras = env.step(actions)
            r = rew.cpu().numpy()
            d = dones.cpu().numpy().astype(bool)
            for i in np.where(alive)[0]:
                buf["q"][i].append(q[i])
                buf["o"][i].append(pobs[i])
                buf["a"][i].append(act[i])
                buf["p"][i].append(rp[i])
                buf["r4"][i].append(rq[i])
                buf["rw"][i].append(r[i])
                if img is not None:
                    buf["img"][i].append(img[i])
                lens[i] += 1
                ret[i] += r[i]
                if d[i]:
                    alive[i] = False
                    time_outs = extras.get("time_outs")
                    ended[i] = "time_out" if time_outs is not None and bool(time_outs[i]) else "terminated"
            if not alive.any():
                break
        for i in range(n):
            arrays = {
                "joint_pos": np.asarray(buf["q"][i], np.float32),
                "policy_obs": np.asarray(buf["o"][i], np.float32),
                "action": np.asarray(buf["a"][i], np.float32),
                "root_pos": np.asarray(buf["p"][i], np.float32),
                "root_quat": np.asarray(buf["r4"][i], np.float32),
                "reward": np.asarray(buf["rw"][i], np.float32),
            }
            if use_cam:
                arrays["image"] = np.asarray(buf["img"][i], np.uint8)
            np.savez(os.path.join(a.out, f"episode_{i:03d}.npz"), **arrays)  # type: ignore[arg-type]
        meta = {
            "task": a.task,
            "checkpoint": a.checkpoint,
            "joint_names": joint_names,
            "action_names": names,
            "policy_obs_dim": int(buf["o"][0][0].shape[0]) if buf["o"][0] else 0,
            "fps": fps,
            "camera": {"width": a.width, "height": a.height} if use_cam else None,
            "episodes": n,
            "episode_len": lens.tolist(),
            "episode_return": [round(float(x), 4) for x in ret],
            "episode_end": ended,
            "wall_s": round(time.time() - t0, 1),
        }
        with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        print("STRANDS_RECORD " + json.dumps({k: meta[k] for k in ("episodes", "episode_len", "fps")}), flush=True)
        genv.close()  # Kit exits the process here
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
