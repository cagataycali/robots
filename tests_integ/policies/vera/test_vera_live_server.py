"""Live integration test for the VERA policy provider (gated; needs a server).

Unlike tests/policies/vera/ (fully mocked, offline), this test talks to a **real**
running ``vera.server.start_vera_server`` — the ``strands-vera-server`` container
or a local server — and validates the on-the-wire handshake + a real chunked
``infer`` (DFoT planner + Jacobian IDM forward pass on a GPU).

Skipped by default. Enable with a running PushT server on :8820::

    # 1. start the server (holds the GPU):
    docker run --rm --gpus all --ipc=host -p 8820:8820 \
        -v "$PWD/vera-ckpts":/ckpts:ro -e VERA_EMBODIMENT=pusht \
        strands-vera-server:latest
    # 2. run the gated test:
    VERA_LIVE=1 hatch run test-integ tests_integ/policies/vera/ -v

Env knobs:
    VERA_LIVE=1            enable (required)
    VERA_LIVE_HOST         server host  (default 127.0.0.1)
    VERA_LIVE_PORT         server port  (default 8820 — PushT)
    VERA_LIVE_EMBODIMENT   expected embodiment for the metadata assertion (default pusht)
"""

from __future__ import annotations

import os

import numpy as np
import pytest

LIVE = os.environ.get("VERA_LIVE", "").lower() in ("1", "true", "yes")
HOST = os.environ.get("VERA_LIVE_HOST", "127.0.0.1")
PORT = int(os.environ.get("VERA_LIVE_PORT", "8820"))
EMBODIMENT = os.environ.get("VERA_LIVE_EMBODIMENT", "pusht")

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="Requires a running vera.server (GPU + checkpoints). Set VERA_LIVE=1 to enable.",
)

# The client transport is import-light (websockets + msgpack); skip cleanly if absent.
pytest.importorskip("websockets", reason="websockets not installed")
pytest.importorskip("msgpack", reason="msgpack not installed")


@pytest.fixture(scope="module")
def client():
    from strands_robots.policies.vera import VeraWebsocketClient

    c = VeraWebsocketClient(host=HOST, port=PORT)
    yield c
    c.close()


def test_handshake_metadata(client):
    """The server advertises a coherent VeraServerConfig on connect."""
    meta = client.get_server_metadata()
    assert meta, "server returned empty metadata"
    assert meta.get("embodiment") == EMBODIMENT
    assert isinstance(meta.get("view_keys"), list) and meta["view_keys"]
    assert int(meta.get("action_dim", 0)) > 0
    assert int(meta.get("context_frames", 0)) >= 1


def test_infer_returns_action_chunk(client):
    """A real context window yields an [H, D] action chunk from the two-stage policy."""
    meta = client.get_server_metadata()
    view_keys = list(meta["view_keys"])
    n_views = len(view_keys)
    # PushT renders ~252; multi-view embodiments use ~128/view. Use a per-view
    # width that keeps sum(view_widths) == concatenated rgb width.
    per_w = 252 if n_views == 1 else 128
    t = int(meta.get("context_frames", 2))
    h = per_w
    ctx = (np.random.rand(t, h, per_w * n_views, 3) * 255).astype(np.uint8)
    req = {
        "context_rgb": ctx,
        "view_keys": view_keys,
        "view_widths": [per_w] * n_views,
        "session_id": "integ-test",
    }
    if meta.get("needs_prompt"):
        req["prompt"] = "push the T to the goal"
    out = client.infer(req)
    action = np.asarray(out["action"])
    assert action.ndim == 2, f"expected [H, D] chunk, got {action.shape}"
    assert action.shape[0] >= 1
    assert action.shape[1] == int(meta["action_dim"])
    assert np.isfinite(action).all()


def test_provider_get_actions_end_to_end():
    """The full VeraPolicy.get_actions path (provider -> server -> action dict)."""
    from strands_robots.policies import create_policy

    policy = create_policy(
        "vera",
        embodiment=EMBODIMENT,
        host=HOST,
        server_port=PORT,
        auto_launch_server=False,
    )
    meta_views = policy._client.get_server_metadata()["view_keys"]
    per_w = 252 if len(meta_views) == 1 else 128
    obs = {k: (np.random.rand(per_w, per_w, 3) * 255).astype(np.uint8) for k in meta_views}
    chunk = policy.get_actions_sync(obs, "push the T to the goal")
    assert isinstance(chunk, list) and chunk
    assert all(isinstance(v, float) for v in chunk[0].values())
    policy.close()
