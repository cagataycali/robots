#!/usr/bin/env python3
"""Phase 4: multi-episode Cosmos3 DROID rollout in MuJoCo + push to HF Hub."""
import os, sys, time
import numpy as np

PORT = 8000
REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "cagataycali/cosmos3-droid-mujoco"
N_EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 48
PUSH = (len(sys.argv) > 4 and sys.argv[4].lower() in ("1","true","push"))

from strands_robots.simulation import Simulation
from strands_robots.policies import create_policy

CUBES = [
    ([0.45, 0.00, 0.05], "pick up the red cube"),
    ([0.50, 0.15, 0.05], "pick up the red cube on the right"),
    ([0.40, -0.15, 0.05], "pick up the red cube on the left"),
]


def main():
    sim = Simulation(tool_name="sim", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", data_config="panda")
    sim.add_object(name="cube", shape="box", position=CUBES[0][0],
                   size=[0.025, 0.025, 0.025], color=[1, 0, 0, 1])
    sim.add_camera(name="wrist", position=[0.45, -0.15, 0.5], target=[0.45, 0, 0.05], width=640, height=480)
    sim.add_camera(name="ext1",  position=[1.0, 0.4, 0.5],   target=[0.45, 0, 0.05], width=640, height=480)
    sim.add_camera(name="ext2",  position=[1.0, -0.4, 0.5],  target=[0.45, 0, 0.05], width=640, height=480)

    policy = create_policy(
        "cosmos3", embodiment="droid", host="localhost", port=PORT,
        observation_mapping={
            "wrist": "observation/wrist_image_left",
            "ext1":  "observation/exterior_image_1_left",
            "ext2":  "observation/exterior_image_2_left",
        },
    )
    policy.set_robot_state_keys([f"joint{i}" for i in range(1, 8)] + ["finger_joint1"])
    print("[policy] ready:", policy.provider_name, policy.embodiment.name)

    rec = sim.start_recording(repo_id=REPO_ID, task=CUBES[0][1], fps=15,
                              push_to_hub=False, vcodec="h264", overwrite=True)
    print("[record] start:", (rec.get("content") or [{}])[0].get("text", rec.get("status"))[:80])

    for ep in range(N_EPISODES):
        pos, instr = CUBES[ep % len(CUBES)]
        sim.reset()
        sim.move_object(name="cube", position=pos)
        policy.reset(seed=ep)
        t0 = time.time()
        res = sim.run_policy(robot_name="arm", policy_object=policy, instruction=instr,
                             n_steps=STEPS, control_frequency=15.0, action_horizon=8, fast_mode=True)
        print(f"[ep {ep}] '{instr}' status={res.get('status')} ({time.time()-t0:.1f}s)")

    stop = sim.stop_recording()
    print("[record] stop:", (stop.get("content") or [{}])[0].get("text", stop.get("status")))

    if PUSH:
        print("[push] pushing to HF Hub:", REPO_ID)
        from strands_robots.dataset_recorder import _get_lerobot_dataset_class
        LeRobotDataset = _get_lerobot_dataset_class()
        ds = LeRobotDataset(REPO_ID)
        ds.push_to_hub(private=False, tags=["cosmos3", "droid", "mujoco", "strands-robots"])
        print("[push] done")


if __name__ == "__main__":
    main()
