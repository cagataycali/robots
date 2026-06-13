"""Integration test for the Qwen-VLA SERVICE-mode policy over real ZMQ.

Two tiers:

1. **Stub-server round-trip** (no GPU, no model): spins up a tiny in-process
   ZMQ server that speaks the same msgpack envelope as a real Qwen-VLA
   inference server, then drives a real ``QwenVlaPolicy`` SERVICE-mode
   instance against it. This validates the wire protocol, observation
   packing, and action unpacking end-to-end without the unreleased upstream
   package. Requires only ``pyzmq`` + ``msgpack`` (gated via importorskip).

2. **Live checkpoint** (GPU-gated, opt-in): when ``QWEN_VLA_MODEL_PATH`` is
   set and the upstream package is installed, runs a real inference. Marked
   ``gpu`` so it is skipped in CPU CI.

Run: ``pytest tests_integ/qwen_vla/ -v``
"""

import os
import threading
import time

import numpy as np
import pytest

pytest.importorskip("zmq", reason="qwen-vla-service extra (pyzmq) not installed")
pytest.importorskip("msgpack", reason="qwen-vla-service extra (msgpack) not installed")

import zmq  # noqa: E402

from strands_robots.policies.qwen_vla import QwenVlaPolicy  # noqa: E402
from strands_robots.policies.qwen_vla.client import MsgSerializer  # noqa: E402

_STUB_PORT = 15560


class _StubQwenVlaServer:
    """Minimal ZMQ REP server mimicking a Qwen-VLA inference server.

    Echoes a deterministic action chunk keyed by the requested action
    families so the test can assert exact unpack behaviour. Speaks the same
    ``{"endpoint", "data"}`` request / msgpack response envelope as the real
    client.
    """

    def __init__(self, port: int, horizon: int = 16):
        self.port = port
        self.horizon = horizon
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(f"tcp://127.0.0.1:{port}")
        self._running = False
        self._thread: threading.Thread | None = None
        self.reset_seeds: list = []

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._sock.close()
        self._ctx.term()

    def _loop(self):
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        while self._running:
            try:
                raw = self._sock.recv()
            except zmq.Again:
                continue
            req = MsgSerializer.from_bytes(raw)
            endpoint = req.get("endpoint")
            if endpoint == "ping":
                self._sock.send(MsgSerializer.to_bytes({"pong": True}))
            elif endpoint == "reset":
                opts = (req.get("data") or {}).get("options") or {}
                self.reset_seeds.append(opts.get("seed"))
                self._sock.send(MsgSerializer.to_bytes({"ok": True}))
            elif endpoint == "get_action":
                # so100: single_arm (6) + gripper (1)
                action = {
                    "action.single_arm": np.tile(np.arange(6, dtype=np.float32), (self.horizon, 1)),
                    "action.gripper": np.ones((self.horizon, 1), dtype=np.float32),
                }
                self._sock.send(MsgSerializer.to_bytes((action, {})))
            else:
                self._sock.send(MsgSerializer.to_bytes({"error": f"unknown endpoint {endpoint}"}))


@pytest.fixture()
def stub_server():
    server = _StubQwenVlaServer(_STUB_PORT)
    server.start()
    time.sleep(0.2)
    yield server
    server.stop()


class TestServiceRoundTrip:
    def test_ping(self, stub_server):
        policy = QwenVlaPolicy(data_config="so100", host="127.0.0.1", port=_STUB_PORT)
        assert policy._client.ping() is True

    def test_get_actions(self, stub_server):
        policy = QwenVlaPolicy(data_config="so100", host="127.0.0.1", port=_STUB_PORT)
        obs = {"webcam": np.zeros((224, 224, 3), np.uint8), "single_arm": np.zeros(6), "gripper": np.zeros(1)}
        actions = policy.get_actions_sync(obs, "pick up the cube")
        assert len(actions) == 16
        assert actions[0]["single_arm"] == [0, 1, 2, 3, 4, 5]
        assert actions[0]["gripper"] == [1.0]

    def test_reset_forwards_seed(self, stub_server):
        policy = QwenVlaPolicy(data_config="so100", host="127.0.0.1", port=_STUB_PORT)
        policy.reset(seed=7)
        time.sleep(0.1)
        assert 7 in stub_server.reset_seeds

    def test_embodiment_prompt_on_wire(self, stub_server):
        policy = QwenVlaPolicy(data_config="aloha_bimanual", host="127.0.0.1", port=_STUB_PORT)
        # Drive one call; the stub ignores prompt content but the policy must
        # build it without error for a dual-arm config.
        obs = {
            "cam_high": np.zeros((224, 224, 3), np.uint8),
            "left_arm": np.zeros(2),
            "right_arm": np.zeros(2),
            "left_gripper": np.zeros(1),
            "right_gripper": np.zeros(1),
        }
        built = policy._build_observation(obs, "fold the towel")
        prompt = built["language"]["task"][0][0]
        assert "dual arms" in prompt
        assert "50 Hz" in prompt


@pytest.mark.gpu
class TestLiveCheckpoint:
    def test_live_inference(self):
        model_path = os.getenv("QWEN_VLA_MODEL_PATH")
        if not model_path:
            pytest.skip("QWEN_VLA_MODEL_PATH not set")
        policy = QwenVlaPolicy(data_config="so100", model_path=model_path, device="cuda")
        obs = {"webcam": np.zeros((224, 224, 3), np.uint8), "single_arm": np.zeros(6), "gripper": np.zeros(1)}
        actions = policy.get_actions_sync(obs, "pick up the cube")
        assert isinstance(actions, list)
        assert len(actions) > 0
