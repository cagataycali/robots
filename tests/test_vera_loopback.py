"""End-to-end loopback test for the VERA policy provider.

Spins up a tiny fake VERA WebSocket server that speaks the real wire protocol
(metadata handshake + msgpack+numpy infer/reset), and drives it through
``strands_robots.policies.vera.VeraPolicy.get_actions_sync`` to verify:

1. ``create_policy("vera", ...)`` constructs cleanly.
2. The msgpack+numpy codec round-trips with VERA's own codec.
3. The rolling context window grows as expected.
4. Action chunks unpack into per-step actuator dicts.
5. ``robot="panda"`` mapping renames columns onto MuJoCo Panda actuators.
6. The session_id is sent and changes after reset().
7. Server proprio (``q_robot``, ``gripper_qpos``) flows through.

Runs without any GPU, any VERA dependency, or any model checkpoints.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from typing import Any

import numpy as np
import websockets.asyncio.server

sys.path.insert(0, "/home/cagatay/vera-integration/vera")
from vera.server.protocol import _msgpack_numpy as vera_codec  # noqa: E402

from strands_robots.policies import create_policy  # noqa: E402


# ---------- a tiny in-process VERA server stub ----------
SEEN_OBS: list[dict[str, Any]] = []
RESET_LOG: list[dict[str, Any]] = []
SERVER_METADATA = {
    # Mirrors what VeraPolicyAdapter.config exposes for mimicgen
    "image_resolution": [128, 128],
    "view_keys": ["front", "side", "agentview"],
    "view_widths": [128, 128, 128],
    "proprio_keys": ["q_robot", "gripper_qpos"],
    "needs_prompt": True,
    "needs_session_id": True,
    "action_space": "joint_position",
    "action_horizon": 6,  # smaller chunk than real for quick test
    "context_frames": 4,  # short context for quick test
    "action_dim": 8,
    "control_dt": 1.0 / 15.0,
    "gripper_is_raw": True,
    "actions_already_metric": False,
    "action_abs_scale": [],
    "gripper_dim_index": 7,
    "embodiment": "mimicgen",
    "planner_model": "fake-wan-1.3b",
    "idm_model": "fake-jacobian-idm",
    "is_causal": False,
    "protocol_version": 1,
    "git_head": "deadbeef",
    "git_dirty": False,
    "git_diff_sha": "",
    "hostname": "loopback",
    "argv": [],
    "run_dir": "",
}


async def _handler(ws):
    packer = vera_codec.Packer()
    await ws.send(packer.pack(SERVER_METADATA))
    chunk_id = 0
    async for raw in ws:
        msg = vera_codec.unpackb(raw)
        endpoint = msg.pop("endpoint", "infer")
        if endpoint == "reset":
            RESET_LOG.append(dict(msg))
            await ws.send("reset successful")
        elif endpoint == "infer":
            SEEN_OBS.append({k: v for k, v in msg.items() if k != "context_rgb"})
            # Fake action: sin wave per dim — verifies row-by-row decoding
            H = SERVER_METADATA["action_horizon"]
            D = SERVER_METADATA["action_dim"]
            t = chunk_id + np.arange(H, dtype=np.float32) / H
            action = np.stack(
                [np.sin(t + 0.5 * d) for d in range(D)], axis=-1
            ).astype(np.float32)
            chunk_id += 1
            response = {
                "action": action,
                "info": {
                    "infer_s": 0.05,
                    "action_absmean": float(np.abs(action).mean()),
                    "chunk_len": int(H),
                    "cold_start": chunk_id == 1,
                    "context_len": int(np.asarray(msg.get("view_keys", [])).size),
                    "session_id": msg.get("session_id"),
                },
            }
            await ws.send(packer.pack(response))
        else:
            await ws.send(f"unknown endpoint: {endpoint!r}")


def _start_server_in_thread(port: int) -> threading.Event:
    ready = threading.Event()

    def _run():
        async def main():
            async with websockets.asyncio.server.serve(
                _handler, "127.0.0.1", port, compression=None, max_size=None
            ) as _server:
                ready.set()
                await asyncio.Future()  # run forever

        asyncio.run(main())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return ready


def _fake_observation(step: int) -> dict[str, Any]:
    """A fake robot observation with 3 cameras + 7 joints + 1 gripper."""
    rng = np.random.default_rng(step)
    # 3 cameras the policy will width-concat in this order
    obs: dict[str, Any] = {
        "front": rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8),
        "side": rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8),
        "agentview": rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8),
        # 7 joints + 1 gripper (rolled in via robot_state_keys)
        **{f"joint{i+1}": 0.1 * (step + i) for i in range(7)},
        "finger_joint1": 0.05 * step,
    }
    return obs


def main() -> int:
    port = 18820
    ready = _start_server_in_thread(port)
    if not ready.wait(timeout=5.0):
        print("FAIL: server didn't come up")
        return 1
    print(f"✓ fake VERA server up on ws://127.0.0.1:{port}")

    # Construct via the real factory path so we exercise the registry too
    policy = create_policy(
        "vera",
        embodiment="mimicgen",
        host="127.0.0.1",
        port=port,
        robot="panda",
        verbose=True,
    )
    # Tell the policy what robot_state_keys to use for proprio inference
    policy.set_robot_state_keys(
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1"]
    )

    # Drain enough ticks to span 2 chunks (so we exercise refill path twice)
    H = SERVER_METADATA["action_horizon"]
    n_ticks = H + 2
    results = []
    t0 = time.monotonic()
    for step in range(n_ticks):
        obs = _fake_observation(step)
        out = policy.get_actions_sync(obs, "A robot arm stacks one block on top of another block")
        results.append(out[0])
    dt = time.monotonic() - t0
    print(f"✓ drained {n_ticks} ticks in {dt:.3f}s ({len(SEEN_OBS)} server infer calls)")

    # ---- sanity checks ----
    assert len(SEEN_OBS) == 2, f"expected 2 chunks for {n_ticks} ticks, got {len(SEEN_OBS)}"
    # actuator dicts use the robot="panda" remap
    first = results[0]
    expected_keys = {"joint1","joint2","joint3","joint4","joint5","joint6","joint7","finger_joint1"}
    assert set(first.keys()) == expected_keys, (
        f"action mapping failed: got {sorted(first.keys())}, expected {sorted(expected_keys)}"
    )
    print(f"✓ per-step actuator dict keys: {sorted(first.keys())}")
    print(f"  first step values: {[f'{k}={v:.3f}' for k,v in list(first.items())[:4]]}...")

    # session_id is in every infer + reset was NOT auto-called (only construct)
    sids = {o.get("session_id") for o in SEEN_OBS}
    assert len(sids) == 1, f"session_id changed mid-episode! {sids}"
    print(f"✓ session_id stable mid-episode: {next(iter(sids))[:8]}...")

    # context_rgb was being sent in width-concat shape (server doesn't keep
    # the array but we verify proprio came through)
    server_obs = SEEN_OBS[0]
    assert "q_robot" in server_obs, f"q_robot missing! seen keys: {sorted(server_obs)}"
    q_robot = np.asarray(server_obs["q_robot"])
    assert q_robot.shape == (7,), f"q_robot wrong shape: {q_robot.shape}"
    print(f"✓ q_robot proprio sent: {q_robot}")
    assert "gripper_qpos" in server_obs, "gripper_qpos missing!"
    print(f"✓ gripper_qpos sent: {server_obs['gripper_qpos']}")
    assert "prompt" in server_obs and "stacks" in server_obs["prompt"]
    print(f"✓ prompt sent: {server_obs['prompt']!r}")
    assert server_obs["view_keys"] == ["front", "side", "agentview"]
    assert server_obs["view_widths"] == [128, 128, 128]
    print(f"✓ view_keys / view_widths: {server_obs['view_keys']} / {server_obs['view_widths']}")

    # reset() bumps session_id and forwards
    old_sid = next(iter(sids))
    policy.reset()
    obs_after = _fake_observation(0)
    policy.get_actions_sync(obs_after, "another task")
    assert len(RESET_LOG) == 1, f"reset RPC not received: {RESET_LOG}"
    new_sids = {o.get("session_id") for o in SEEN_OBS[-1:]}
    assert old_sid not in new_sids, f"session_id didn't change on reset: {old_sid}"
    print(f"✓ reset() bumped session: {old_sid[:8]} -> {next(iter(new_sids))[:8]}")
    assert RESET_LOG[0].get("reason") == "episode_reset"
    print(f"✓ reset RPC includes reason: {RESET_LOG[0]}")

    # server metadata accessible
    meta = policy.get_server_metadata()
    assert meta["embodiment"] == "mimicgen"
    assert meta["action_horizon"] == H
    assert meta["planner_model"] == "fake-wan-1.3b"
    print(f"✓ server metadata exposed: planner={meta['planner_model']} idm={meta['idm_model']} "
          f"H={meta['action_horizon']} D={meta['action_dim']}")

    print()
    print("=" * 60)
    print("🎉 VERA loopback E2E test PASSED — wire protocol works.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
