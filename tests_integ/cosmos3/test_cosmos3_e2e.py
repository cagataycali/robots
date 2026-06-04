"""Integration test: real OpenPI wire round-trip through Cosmos3Policy.

Spins up an OpenPI ``WebsocketPolicyServer`` with a *fake* policy that returns a
deterministic ``(32, 8)`` DROID action chunk, then drives it through the real
:class:`strands_robots.policies.cosmos3.Cosmos3Policy` service-mode client. This
exercises the actual msgpack+NumPy wire path (encode obs -> server -> decode
action -> per-step dicts) without needing a GPU or the 16B checkpoint.

Gated on ``openpi-client`` + ``openpi-server`` (the ``cosmos3-service`` extra).
Run via ``hatch run test-integ`` / ``pytest tests_integ/cosmos3/``.

The full GPU verification (real ``nvidia/Cosmos3-Nano-Policy-DROID`` -> (32, 8)
chunk on an L40S, then a 3-episode MuJoCo rollout pushed to
``cagataydev/cosmos3-droid-mujoco``) was done out-of-band; this keeps a
CPU-only, deterministic regression of the wire contract in CI.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

openpi_client = pytest.importorskip(
    "openpi_client", reason="openpi-client not installed - pip install 'strands-robots[cosmos3-service]'"
)
pytest.importorskip("openpi_client.base_policy", reason="openpi-client too old (no base_policy)")
# Server side ships in openpi-server (or full openpi). Skip if absent.
try:
    from openpi_server.websocket_policy_server import WebsocketPolicyServer
except ModuleNotFoundError:  # pragma: no cover
    try:
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    except ModuleNotFoundError:
        WebsocketPolicyServer = None

pytestmark = pytest.mark.skipif(
    WebsocketPolicyServer is None,
    reason="openpi WebsocketPolicyServer not installed",
)

from openpi_client.base_policy import BasePolicy  # noqa: E402

from strands_robots.policies import create_policy  # noqa: E402

_CHUNK = np.arange(32 * 8, dtype=np.float32).reshape(32, 8) * 0.01


def _free_port() -> int:
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _FakeDroidPolicy(BasePolicy):
    """Echoes a fixed (32, 8) chunk and asserts the obs wire shape."""

    def __init__(self):
        self.last_obs = None

    def infer(self, obs):
        self.last_obs = obs
        assert "prompt" in obs
        assert obs["observation/joint_position"].shape == (1, 7)
        assert obs["observation/gripper_position"].shape == (1, 1)
        return {"action": _CHUNK}

    def reset(self):
        pass


@pytest.fixture()
def server_port():
    port = _free_port()
    server = WebsocketPolicyServer(policy=_FakeDroidPolicy(), host="localhost", port=port, metadata={"ok": True})
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(1.0)  # let the event loop bind
    yield port


def _obs():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    obs = {
        "observation/wrist_image_left": img,
        "observation/exterior_image_1_left": img,
        "observation/exterior_image_2_left": img,
    }
    for i in range(7):
        obs[f"joint_{i}"] = 0.1 * i
    obs["gripper"] = 0.5
    return obs


def test_real_wire_roundtrip_through_cosmos3_policy(server_port):
    policy = create_policy("cosmos3", embodiment="droid", host="localhost", port=server_port)
    policy.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])

    assert policy._client.get_server_metadata() == {"ok": True}

    chunk = policy.get_actions_sync(_obs(), "pick up the cube")
    assert len(chunk) == 32
    assert set(chunk[0]) == {f"joint_{i}" for i in range(7)} | {"gripper"}
    # column 0 of row 1 = 8 * 0.01 == 0.08 (verifies ordering + naming end-to-end)
    assert abs(chunk[1]["joint_0"] - 0.08) < 1e-6


def test_robot_panda_mapping_over_real_wire(server_port):
    policy = create_policy("cosmos3", embodiment="droid", host="localhost", port=server_port, robot="panda")
    policy.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    chunk = policy.get_actions_sync(_obs(), "go")
    assert set(chunk[0]) == {f"joint{i}" for i in range(1, 8)} | {"finger_joint1"}
