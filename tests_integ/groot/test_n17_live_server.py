"""Live-server integration test for GR00T N1.7.

Unlike test_groot_integration.py (which spins up a local N1.6 server), this
test talks to a **pre-running** N1.7 server on the LAN / localhost.  That
makes it cheap: it can run from a Mac dev box pointed at a Thor / DGX / EC2
GPU host without needing CUDA or the ~10GB model weights locally.

Enable with:

    GROOT_LIVE_SERVER=1 \
    GROOT_SERVER_HOST=192.168.1.151 \
    GROOT_SERVER_PORT=5555 \
    hatch run test-integ tests_integ/groot/test_n17_live_server.py -v

Assertions in this file prove end-to-end connectivity against a real N1.7
server: (1) dict-form ``ModalityConfig`` decode, (2) forward-compat unknown
fields, (3) correct request envelope, (4) tuple response unpacking, and (5)
the REAL_G1 data config schema matches what the server actually reports.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

LIVE = os.environ.get("GROOT_LIVE_SERVER", "").lower() in ("1", "true", "yes")
HOST = os.environ.get("GROOT_SERVER_HOST", "localhost")
PORT = int(os.environ.get("GROOT_SERVER_PORT", "5555"))

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="Requires a pre-running GR00T N1.7 server. Set GROOT_LIVE_SERVER=1 to enable.",
)

# Skip cleanly if optional deps are missing.
msgpack = pytest.importorskip("msgpack")
zmq = pytest.importorskip("zmq")

from strands_robots.policies.groot.client import Gr00tInferenceClient  # noqa: E402
from strands_robots.policies.groot.data_config import (  # noqa: E402
    DATA_CONFIG_MAP,
    ModalityConfig,
)


@pytest.fixture(scope="module")
def client():
    c = Gr00tInferenceClient(host=HOST, port=PORT, timeout_ms=30_000)
    yield c


def _build_real_g1_obs() -> dict:
    """Realistic REAL_G1 observation (all-zeros SVDs-crash inside the server)."""
    identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    left_eef = np.concatenate([[0.15, 0.25, 0.2], identity_rot6d]).astype(np.float32)
    right_eef = np.concatenate([[0.15, -0.25, 0.2], identity_rot6d]).astype(np.float32)
    video = np.random.randint(0, 255, (1, 2, 256, 256, 3), dtype=np.uint8)
    return {
        "video": {"ego_view": video},
        "state": {
            "left_wrist_eef_9d": left_eef[np.newaxis, np.newaxis, :],
            "right_wrist_eef_9d": right_eef[np.newaxis, np.newaxis, :],
            "left_hand": np.zeros((1, 1, 7), dtype=np.float32) + 0.01,
            "right_hand": np.zeros((1, 1, 7), dtype=np.float32) + 0.01,
            "left_arm": np.array([[[0.1, -0.3, 0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
            "right_arm": np.array([[[0.1, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
            "waist": np.zeros((1, 1, 3), dtype=np.float32),
        },
        "language": {"annotation.human.task_description": [["pick up the object"]]},
    }


def test_ping(client):
    assert client.ping() is True


def test_modality_config_decode_roundtrip(client):
    """Verify all 4 modalities decode to ModalityConfig instances.

    Exercises the N1.7 dict-form ``as_json`` decode AND the forward-compat
    fallback that drops unknown fields (``sin_cos_embedding_keys`` etc.).
    """
    resp = client.call_endpoint("get_modality_config")
    assert set(resp.keys()) == {"video", "state", "action", "language"}
    for key, cfg in resp.items():
        assert isinstance(cfg, ModalityConfig), f"{key}: got {type(cfg).__name__}"
        assert isinstance(cfg.delta_indices, list)
        assert isinstance(cfg.modality_keys, list)


def test_modality_config_matches_real_g1_local(client):
    """The server's REAL_G1 schema must match our local `unitree_g1_real` config.

    If NVIDIA ships a new REAL_G1 checkpoint with different obs/action keys,
    this test catches the drift.
    """
    server_cfg = client.call_endpoint("get_modality_config")
    local_cfg = DATA_CONFIG_MAP["unitree_g1_real"]

    # Video: same ego_view key and same [-20, 0] delta indices
    assert server_cfg["video"].delta_indices == local_cfg.observation_indices
    assert [f"video.{k}" for k in server_cfg["video"].modality_keys] == local_cfg.video_keys

    # State: same keys (order need not match)
    assert set(f"state.{k}" for k in server_cfg["state"].modality_keys) == set(local_cfg.state_keys)

    # Action horizon: same size
    assert server_cfg["action"].delta_indices == local_cfg.action_indices
    assert set(f"action.{k}" for k in server_cfg["action"].modality_keys) == set(local_cfg.action_keys)


def test_get_action_real_inference(client):
    """Send a real REAL_G1 obs and verify the action chunk shape.

    Exercises the full pipeline: request envelope wrapping, tuple response
    unpacking, msgpack-over-ZMQ transport, and the actual diffusion-head
    denoising on the GPU.
    """
    obs = _build_real_g1_obs()
    t0 = time.perf_counter()
    actions = client.get_action(obs)
    dt_ms = (time.perf_counter() - t0) * 1000

    assert "navigate_command" in actions, "REAL_G1 should emit navigate_command"
    assert "base_height_command" in actions
    nav = np.asarray(actions["navigate_command"])
    # Expected horizon for REAL_G1: 40 steps, 3 dims (vx, vy, vyaw), B=1.
    assert nav.shape == (1, 40, 3), f"navigate_command shape={nav.shape}"
    assert nav.dtype == np.float32

    # Sanity: no NaN / inf in actions (common sign of server-side numerical blow-up)
    for key, arr in actions.items():
        a = np.asarray(arr)
        assert np.isfinite(a).all(), f"{key} contains NaN/Inf — server numerical issue?"

    # Loose latency sanity — warm inference is sub-500ms on Thor but cold can
    # be much higher (bfloat16 weight upload). Just warn, don't fail.
    print(f"\nREAL_G1 inference latency: {dt_ms:.0f}ms  (informational)")
