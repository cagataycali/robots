"""Experiment runner: N episodes x arms x tasks on the shared so101 MuJoCo scene, fixed seeds, LeRobot v3 recording.

Arms: scripted, random, laya-english, laya-multilingual, laya-typed-decisions (gated question profile, NO gate applied
so H1 measures raw actuation and H2 is scored offline from the recorded probabilities).
Per episode: seed = 1000 * task_index + episode -> cube xy (same across arms), robot.reset() to the rest pose,
run_policy(n_steps, stop_when=success), per-tick records with distance / cube z / contacts / Laya answers.
Output: results/<arm>_<task>.jsonl (one episode per line), lerobot/<arm>_<task>/ (LeRobot v3, when --record).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from laya_scene import (  # noqa: E402
    FPS,
    TASKS,
    RandomPrimitivePolicy,
    ScriptedPrimitivePolicy,
    World,
    build,
    randomized_cube_xy,
)

from strands_robots.policies.laya import LayaPolicy  # noqa: E402

HERE = Path(__file__).resolve().parent
ARMS = ("scripted", "random", "english", "multilingual", "typed-decisions")
LAYA_ARMS = ("english", "multilingual", "typed-decisions")


def make_policy(arm: str, task: str, world: World, device: str) -> object:
    if arm == "scripted":
        return ScriptedPrimitivePolicy(world, task)
    if arm == "random":
        return RandomPrimitivePolicy(world, 0)
    policy = LayaPolicy(model=arm, device=device, questions_profile="joint_direction_size_gated", confidence_gate=None)
    policy.set_world_reader(world.read)
    return policy


def summarize(ticks: list[dict], success: bool, steps: int) -> dict:
    lat = [t["latency_ms"] for t in ticks[1:]] or [0.0]
    applied = [t["applied"]["joint"] for t in ticks]
    return {
        "success": success,
        "steps": steps,
        "steps_to_success": steps if success else None,
        "min_distance_m": min(t["distance_after"] for t in ticks) if ticks else None,
        "final_distance_m": ticks[-1]["distance_after"] if ticks else None,
        "cube_max_z_m": max(t["cube_z"] for t in ticks) if ticks else None,
        "finger_contact_any": any(t["finger_contact"] for t in ticks),
        "hold_fraction": applied.count("none") / max(len(applied), 1),
        "gripper_fraction": applied.count("gripper") / max(len(applied), 1),
        "latency_ms_p50": statistics.median(lat),
        "latency_ms_p95": sorted(lat)[max(int(0.95 * len(lat)) - 1, 0)],
        "latency_ms_first": ticks[0]["latency_ms"] if ticks else None,
    }


def run(arm: str, task: str, episodes: int, steps: int, device: str, record: bool, out_dir: Path, repo_id: str) -> None:
    robot = build(cameras=record)
    world = World(robot)
    policy = make_policy(arm, task, world, device)
    out_path = out_dir / "results" / f"{arm}_{task}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["episode"])
    task_index = list(TASKS).index(task)
    if arm in LAYA_ARMS:
        # warm-up: model load happens here, not inside the first recorded tick
        t0 = time.perf_counter()
        policy.get_actions_sync({k: 0.0 for k in world.ctrl_bounds}, TASKS[task])
        print(f"[{arm}/{task}] laya warm-up {time.perf_counter() - t0:.1f}s", flush=True)
        policy.reset()
    if record:
        rec_root = out_dir / "lerobot" / f"{arm}_{task}"
        res = robot.start_recording(
            repo_id=repo_id,
            task=TASKS[task],
            fps=FPS,
            root=str(rec_root),
            push_to_hub=False,
            cameras=["scene", "wrist"],
        )
        assert res["status"] == "success", res
    with out_path.open("a") as fh:
        for ep in range(episodes):
            if ep in done:
                continue
            seed = 1000 * task_index + ep
            rng = np.random.default_rng(seed)
            x, y = randomized_cube_xy(rng)
            robot.reset()
            assert robot.move_object("cube", position=[x, y, 0.01])["status"] == "success"
            policy.reset(seed=seed)
            ticks: list[dict] = []

            def observe(ev, _p=policy, _t=ticks):
                if type(ev).__name__ == "RunPolicyStep" and _p.last_tick is not None:
                    t = dict(_p.last_tick)
                    t["distance_after"] = world.distance()
                    t["cube_z"] = world.cube_xyz()[2]
                    t["contacts"] = world.contacts()
                    t["finger_contact"] = world.finger_contact()
                    t["success"] = world.success(task)
                    _t.append(t)

            start_distance = world.distance()
            t0 = time.perf_counter()
            ran = robot.run_policy(
                robot_name="so101",
                policy_object=policy,
                instruction=TASKS[task],
                n_steps=steps,
                control_frequency=FPS,
                fast_mode=True,
                observer=observe,
                stop_when=lambda eng, _task=task: world.success(_task),
            )
            wall = time.perf_counter() - t0
            info = ran["content"][1]["json"] if ran.get("status") == "success" else {}
            success = world.success(task)
            saved = robot.save_episode() if record else None
            episode = {
                "arm": arm,
                "task": task,
                "episode": ep,
                "seed": seed,
                "cube_xy": [x, y],
                "start_distance_m": start_distance,
                "run_status": ran.get("status"),
                "steps_used": info.get("steps_used"),
                "stopped_reason": info.get("stopped_reason"),
                "action_errors": info.get("action_errors"),
                "wall_s": wall,
                "saved": (saved or {}).get("status"),
                "summary": summarize(ticks, success, info.get("steps_used") or len(ticks)),
                "ticks": ticks,
            }
            fh.write(json.dumps(episode) + "\n")
            fh.flush()
            s = episode["summary"]
            print(
                f"[{arm}/{task}] ep {ep:02d} seed {seed} success={success} steps={s['steps']} "
                f"min_d={s['min_distance_m']:.3f} hold={s['hold_fraction']:.2f} lat_p50={s['latency_ms_p50']:.1f}ms wall={wall:.1f}s",
                flush=True,
            )
    if record:
        print(f"[{arm}/{task}] stop_recording:", str(robot.stop_recording())[:300], flush=True)
    robot.destroy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--repo-id", default="cagataydev/laya-so101-mujoco-20260928")
    args = ap.parse_args()
    for arm in args.arms:
        for task in args.tasks:
            run(arm, task, args.episodes, args.steps, args.device, args.record, Path(args.out), args.repo_id)


if __name__ == "__main__":
    main()
