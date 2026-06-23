#!/usr/bin/env python3
"""Agent-driven data collection + streaming read-back (the physical-AI loop).

Demonstrates the full strands-robots × LeRobot streaming/bucket integration
described in reports/STREAMING_DATA_LOOP_DEEP_DIVE.md:

  1. A Strands Agent with Robot("so100") as a tool composes a scene
     (cubes + cameras) and runs a policy — recording a LeRobotDataset.
  2. We then STREAM the recorded dataset back with StreamingDatasetReader
     (Phase 3 of the data loop) — no full re-materialization.

Run:
    MUJOCO_GL=cgl python examples/06_agent_collect_and_stream.py

Dependencies:
    pip install "strands-robots[sim-mujoco,lerobot]" strands-agents
    AWS credentials for Bedrock (or any strands-agents model provider).
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl")  # macOS offscreen GL

from strands import Agent

from strands_robots import Robot, StreamingDatasetReader

DATASET_ROOT = "/tmp/strands_agent_dataset"
REPO_ID = "local/agent_demo"

# ── 1. Robot() is a Strands AgentTool — hand it to an Agent ───────────────
sim = Robot("so100", mesh=False)
agent = Agent(tools=[sim])

# ── 2. One natural-language prompt drives scene + cameras + policy + record ─
result = agent(
    f"Create a world with the so100 robot. Add a small red cube at "
    f"[0.2, 0.0, 0.05] and a blue cube at [0.25, 0.05, 0.05]. Add a front "
    f"camera looking at the cubes. Then START RECORDING a LeRobot dataset "
    f"with repo_id='{REPO_ID}', root='{DATASET_ROOT}', fps=30, overwrite=True, "
    f"task='pick up the red cube'. Run the mock policy for 60 steps with "
    f"instruction 'pick up the red cube'. Finally STOP RECORDING to save the "
    f"episode to disk."
)
print("\n=== AGENT RESULT ===")
print(result)

# ── 3. Stream the recorded dataset back (Phase 3) ─────────────────────────
# StreamingLeRobotDataset decodes video for ANY dataset with camera keys, so
# torchcodec must be importable to stream a video dataset. On macOS that means
# a matching torch/torchcodec/ffmpeg trio (deep-dive Appendix C). When torchcodec
# is unavailable we fall back to demonstrating the streaming API on a
# proprio-only (no-camera) dataset — the torchcodec-free path (App. C.2).
print("\n=== STREAMING READ-BACK (Phase 3) ===")


def _torchcodec_ok() -> bool:
    try:
        import torchcodec  # noqa: F401
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        return True
    except Exception:
        return False


if _torchcodec_ok():
    reader = StreamingDatasetReader.open(
        REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps={
            "observation.images.front_camera": [-0.0667, 0.0],
            "observation.state": [-0.0667, -0.0333, 0.0],
            "action": [0.0, 0.0333, 0.0667],
        },
        shuffle=False,
        streaming=True,
    )
    print(f"episodes={reader.num_episodes} frames={reader.num_frames} fps={reader.fps}")
    n = 0
    for frame in reader:
        if n < 3:
            cam = [k for k in frame if k.startswith("observation.images.")]
            print(f"  frame {n}: state{tuple(frame['observation.state'].shape)} "
                  f"action{tuple(frame['action'].shape)} cams={cam}")
        n += 1
        if n >= 5:
            break
    print(f"\n✅ Streamed {n} windowed frames (incl. video) from the agent dataset.")
else:
    print("torchcodec unavailable → proving the streaming API on a proprio-only "
          "dataset (no camera keys, no video decode — deep-dive App. C.2).")
    import numpy as np
    from strands_robots.dataset_recorder import DatasetRecorder

    pr_root, pr_repo = "/tmp/strands_proprio_demo", "local/proprio_demo"
    rec = DatasetRecorder.create(
        repo_id=pr_repo, root=pr_root, fps=30, robot_type="so100",
        joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
        task="reach", use_videos=False, camera_keys=None,
    )
    for ep in range(2):
        for t in range(40):
            obs = {f"j{i+1}": float(np.sin(t * 0.1 + i)) for i in range(6)}
            act = {f"j{i+1}": float(np.cos(t * 0.1 + i)) for i in range(6)}
            rec.add_frame(obs, act, task="reach")
        rec.save_episode()
    rec.finalize()

    reader = StreamingDatasetReader.open(
        pr_repo, root=pr_root,
        delta_timestamps={
            "observation.state": [-0.0667, -0.0333, 0.0],
            "action": [0.0, 0.0333, 0.0667],
        },
        shuffle=False, streaming=True,
    )
    print(f"episodes={reader.num_episodes} frames={reader.num_frames} fps={reader.fps}")
    n = 0
    for frame in reader:
        if n < 3:
            pads = [k for k in frame if k.endswith("_is_pad")]
            print(f"  frame {n}: state{tuple(frame['observation.state'].shape)} "
                  f"action{tuple(frame['action'].shape)} pad={pads}")
        n += 1
        if n >= 5:
            break
    print(f"\n✅ Streamed {n} windowed frames (proprio-only, torchcodec-free).")

print("   (delta windows + *_is_pad masks applied; nothing re-downloaded)")

# ── 4. Optional: push to an HF Storage Bucket (Phase 1/2) ─────────────────
# recorder.sync_to_bucket("your-org/robot-bucket")  # mutable, Xet-deduped
print("\n💡 To dump to a mutable HF Storage Bucket (Phase 1/2):")
print("   stop_recording(..., bucket='your-org/robot-bucket')  # see deep-dive §2.4")
