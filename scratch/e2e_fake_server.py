"""Live E2E: real Cosmos3Policy + real OpenPI client/server (fake policy, no GPU).

Spins up an OpenPI WebsocketPolicyServer with a fake policy that echoes a
deterministic (32,8) DROID action chunk, then drives it through the actual
strands_robots Cosmos3Policy. Proves the real wire path end-to-end.
"""
import sys, threading, time
import numpy as np

sys.path.insert(0, ".")  # strands_robots

from openpi_server.websocket_policy_server import WebsocketPolicyServer
from openpi_client.base_policy import BasePolicy

PORT = 8777
CHUNK = np.arange(32 * 8, dtype=np.float32).reshape(32, 8) * 0.01


class FakeDroidPolicy(BasePolicy):
    def infer(self, obs):
        # echo back what the client sent + a fixed chunk so we can assert the wire
        assert "prompt" in obs, obs.keys()
        assert "observation/joint_position" in obs, obs.keys()
        assert obs["observation/joint_position"].shape == (1, 7)
        return {"action": CHUNK, "echo_prompt": obs["prompt"]}

    def reset(self):
        pass


def serve():
    WebsocketPolicyServer(policy=FakeDroidPolicy(), host="0.0.0.0", port=PORT, metadata={"ok": True}).serve_forever()


def main():
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    time.sleep(2.0)

    from strands_robots.policies import create_policy

    policy = create_policy("cosmos3", embodiment="droid", host="localhost", port=PORT)
    policy.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])

    img = np.zeros((360, 640, 3), dtype=np.uint8)
    obs = {
        "observation/wrist_image_left": img,
        "observation/exterior_image_1_left": img,
        "observation/exterior_image_2_left": img,
    }
    for i in range(7):
        obs[f"joint_{i}"] = 0.1 * i
    obs["gripper"] = 0.5

    chunk = policy.get_actions_sync(obs, "pick up the cube")
    print("[e2e] chunk len:", len(chunk))
    print("[e2e] step0 keys:", sorted(chunk[0].keys()))
    print("[e2e] step0:", {k: round(v, 4) for k, v in chunk[0].items()})
    assert len(chunk) == 32, "expected 32 steps"
    assert set(chunk[0]) == {f"joint_{i}" for i in range(7)} | {"gripper"}
    # check values match the fake chunk (column-named correctly)
    assert abs(chunk[0]["joint_0"] - 0.0) < 1e-6
    assert abs(chunk[1]["joint_0"] - 0.08) < 1e-6  # row1 col0 = 8*0.01
    print("[e2e] metadata:", policy._client.get_server_metadata())
    print("[e2e] ✅ real client↔server roundtrip through Cosmos3Policy PASSED")


if __name__ == "__main__":
    main()
