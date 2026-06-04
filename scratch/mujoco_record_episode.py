#!/usr/bin/env python3
"""Phase 4: live Cosmos3-Nano-Policy-DROID rollout in MuJoCo + LeRobotDataset record.

Spins up a Franka/Panda (DROID-compatible 7-DOF) arm in MuJoCo, adds a cube and
three cameras matching the DROID embodiment views, then rolls out the *real*
Cosmos 3 policy (via the running robolab WebSocket server) and records the
episode to a LeRobotDataset (parquet + per-camera MP4), optionally pushing to HF.

Usage:
  python scratch/mujoco_record_episode.py [port] [repo_id] [n_steps] [push]
"""
import os, sys, time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
REPO_ID = sys.argv[2] if len(sys.argv) > 2 else "local/cosmos3_droid_pick"
N_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
PUSH = (len(sys.argv) > 4 and sys.argv[4].lower() in ("1", "true", "push"))
INSTRUCTION = "pick up the red cube"

from strands_robots.simulation import Simulation
from strands_robots.policies import create_policy


def main():
    sim = Simulation(tool_name="sim", mesh=False)
    print("[sim] create_world"); print(sim.create_world().get("status"))
    # Franka Panda = 7-DOF DROID-compatible arm (alias oxe_droid / libero_panda).
    print("[sim] add_robot panda"); print(sim.add_robot(name="arm", data_config="panda").get("status"))
    sim.add_object(name="cube", shape="box", position=[0.45, 0.0, 0.05], size=[0.025, 0.025, 0.025], color=[1, 0, 0, 1])
    # DROID-style 3 cameras. Names map to the server's OpenPI image keys.
    sim.add_camera(name="wrist", position=[0.45, -0.15, 0.5], target=[0.45, 0.0, 0.05], width=640, height=480)
    sim.add_camera(name="ext1",  position=[1.0, 0.4, 0.5],   target=[0.45, 0.0, 0.05], width=640, height=480)
    sim.add_camera(name="ext2",  position=[1.0, -0.4, 0.5],  target=[0.45, 0.0, 0.05], width=640, height=480)
    print("[sim] cameras added")

    # Build the real Cosmos 3 policy (service mode → robolab WS server).
    policy = create_policy(
        "cosmos3",
        embodiment="droid",
        host="localhost",
        port=PORT,
        observation_mapping={
            "wrist": "observation/wrist_image_left",
            "ext1":  "observation/exterior_image_1_left",
            "ext2":  "observation/exterior_image_2_left",
        },
    )
    # Panda obs uses joint1..joint7 + finger_joint1 (gripper). Tell the policy
    # which keys are the 7 joints and which is the gripper so joint_pos state maps.
    policy.set_robot_state_keys([f"joint{i}" for i in range(1, 8)] + ["finger_joint1"])
    print("[policy] cosmos3 ready:", policy.provider_name, policy.embodiment.name, policy.action_space)
    print("[policy] server metadata:", policy._client.get_server_metadata())

    # Start LeRobotDataset recording (parquet + per-cam MP4). 15 fps = DROID fps.
    rec = sim.start_recording(
        repo_id=REPO_ID, task=INSTRUCTION, fps=15, push_to_hub=PUSH,
        vcodec="h264", overwrite=True,
    )
    print("[record] start:", (rec.get("content") or [{}])[0].get("text", rec.get("status")))

    # Roll out the real policy in sim. control_frequency=15 to match the policy.
    t0 = time.time()
    res = sim.run_policy(
        robot_name="arm",
        policy_object=policy,
        instruction=INSTRUCTION,
        n_steps=N_STEPS,
        control_frequency=15.0,
        action_horizon=8,
        fast_mode=True,
    )
    dt = time.time() - t0
    print(f"[run] status={res.get('status')} ({dt:.1f}s):", (res.get("content") or [{}])[0].get("text", "")[:300])

    stop = sim.stop_recording()
    print("[record] stop:", (stop.get("content") or [{}])[0].get("text", stop.get("status")))
    print("[done] dataset repo_id:", REPO_ID, "push_to_hub:", PUSH)


if __name__ == "__main__":
    main()
