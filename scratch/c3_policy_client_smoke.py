#!/usr/bin/env python3
"""Phase 1 smoke: drive the Cosmos3-Nano-Policy-DROID robolab server.

Builds an OpenPI-protocol observation dict from the checked-in DROID example
(first frame of each camera + a 7-DOF joint state), sends it to the running
robolab WebSocket policy server, and dumps the returned action chunk.

Run AFTER the server is up:
  python -m cosmos_framework.scripts.action_policy_server_robolab \
      --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000
"""
import sys, json
import numpy as np

HOST = "localhost"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
PROMPT = "Pick up the object and move it."

def load_frame(name):
    return np.load(f"scratch/droid_frames/{name}.npy")  # [360,640,3] uint8

def build_joint_state():
    # lerobot v3 flattens 7-DOF joints into 7 consecutive scalar rows.
    import pandas as pd
    df = pd.read_parquet(
        "../cosmos/cookbooks/cosmos3/generator/action/assets/"
        "droid_lerobot_example/data/chunk-000/file-000.parquet"
    )
    jp = df["observation.state.joint_positions"].iloc[:7].to_numpy(dtype=np.float32)
    grip = np.float32(df["observation.state.gripper_position"].iloc[0])
    return jp.reshape(1, 7), np.array([[grip]], dtype=np.float32)  # [1,7],[1,1]

def main():
    from openpi_client import websocket_client_policy

    joint, gripper = build_joint_state()
    obs = {
        "prompt": PROMPT,
        # RoBoArena 3-cam path (server auto-composes wrist-on-top + 2 exteriors)
        "observation/wrist_image_left": load_frame("wrist"),
        "observation/exterior_image_1_left": load_frame("ext1"),
        "observation/exterior_image_2_left": load_frame("ext2"),
        # joint_pos action space needs joint(7) + gripper
        "observation/joint_position": joint,
        "observation/gripper_position": gripper,
    }
    print("[client] obs keys:", list(obs.keys()))
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")

    client = websocket_client_policy.WebsocketClientPolicy(host=HOST, port=PORT)
    print(f"[client] connected to ws://{HOST}:{PORT}; server metadata:", client.get_server_metadata())

    print("[client] infer() ...")
    result = client.infer(obs)
    print("[client] result keys:", list(result.keys()))
    action = np.asarray(result["action"])
    print(f"[client] action chunk shape: {action.shape} dtype={action.dtype}")
    print(f"[client] action[0]: {action[0].tolist()}")
    print(f"[client] action[-1]: {action[-1].tolist()}")
    print(f"[client] per-dim min/max:")
    for d in range(action.shape[-1]):
        col = action[:, d]
        print(f"    dim{d}: min={col.min():.4f} max={col.max():.4f}")
    if "video" in result:
        v = np.asarray(result["video"]); print(f"[client] video: {v.shape} {v.dtype}")
    # dump for provider reference
    np.save("scratch/c3_action_chunk.npy", action)
    with open("scratch/c3_action_chunk.json", "w") as f:
        json.dump({"shape": list(action.shape), "dtype": str(action.dtype),
                   "first": action[0].tolist(), "last": action[-1].tolist()}, f, indent=2)
    print("[client] saved scratch/c3_action_chunk.{npy,json}")

if __name__ == "__main__":
    main()
