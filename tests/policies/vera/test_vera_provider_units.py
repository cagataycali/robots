"""Unit coverage for VERA provider frame coercion and action-column binding.

These exercise the embodiment-agnostic plumbing of :mod:`strands_robots.policies.vera.provider`
that does not need a running VERA server or GPU:

* ``_to_uint8_frame`` -- camera-frame dtype/shape coercion (float scaling, batch
  squeeze, integer clamping, shape rejection).
* ``_resize_frame`` -- the no-op fast path and the PIL-less numpy fallback.
* ``VeraPolicy._action_column_names`` (via the public ``get_actions`` chunk
  binding) -- joint-name binding, trailing-gripper extras, and the warn-once
  unbound fallback that emits ``action_i`` keys.
* ``VeraPolicy.close`` -- fail-soft client close plus managed-runner stop.

All assertions are on observable outputs (returned arrays/dicts, emitted log
records, recorded runner calls), not internal state.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import numpy as np
import pytest

from strands_robots.policies.vera.provider import (
    VeraPolicy,
    _resize_frame,
    _to_uint8_frame,
)


class TestToUint8Frame:
    """``_to_uint8_frame`` coerces arbitrary camera values to (H, W, 3) uint8."""

    def test_float_frame_scaled_to_0_255(self):
        frame = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)  # (1, 1, 3)
        out = _to_uint8_frame(frame)
        assert out.dtype == np.uint8
        assert out.shape == (1, 1, 3)
        assert list(out[0, 0]) == [0, 127, 255]

    def test_float_frame_clipped_before_scaling(self):
        # Values outside [0, 1] clamp rather than wrap around.
        frame = np.array([[[-1.0, 2.0, 0.25]]], dtype=np.float64)
        out = _to_uint8_frame(frame)
        assert list(out[0, 0]) == [0, 255, 63]

    def test_batch_dim_is_squeezed(self):
        frame = np.zeros((1, 4, 5, 3), dtype=np.uint8)
        out = _to_uint8_frame(frame)
        assert out.shape == (4, 5, 3)

    def test_integer_frame_clamped_to_uint8(self):
        frame = np.array([[[300, -5, 128]]], dtype=np.int16)
        out = _to_uint8_frame(frame)
        assert out.dtype == np.uint8
        assert list(out[0, 0]) == [255, 0, 128]

    def test_uint8_frame_returned_contiguous(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)[::1]
        out = _to_uint8_frame(frame)
        assert out.dtype == np.uint8
        assert out.flags["C_CONTIGUOUS"]

    def test_bad_shape_raises_valueerror(self):
        with pytest.raises(ValueError, match=r"must be \(H, W, 3\)"):
            _to_uint8_frame(np.zeros((4, 5), dtype=np.uint8))


class TestResizeFrame:
    """``_resize_frame`` squares each view to the planner's per-view width."""

    def test_already_square_is_identity(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        out = _resize_frame(frame, 8)
        assert out is frame

    def test_numpy_fallback_when_pil_unavailable(self, monkeypatch):
        # Force `from PIL import Image` to raise so the numpy nearest-neighbour
        # branch runs; it must still produce a square (width, width, 3) frame.
        monkeypatch.setitem(sys.modules, "PIL", None)
        frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        out = _resize_frame(frame, 5)
        assert out.shape == (5, 5, 3)
        assert out.dtype == np.uint8
        assert out.flags["C_CONTIGUOUS"]


class _FakeClient:
    """Scriptable VeraWebsocketClient stand-in (no socket)."""

    def __init__(self, metadata, action_chunk, *, raise_on_close=False):
        self._meta = metadata
        self._chunk = np.asarray(action_chunk, dtype=np.float32)
        self._raise_on_close = raise_on_close
        self.closed = False

    def get_server_metadata(self):
        return dict(self._meta)

    def infer(self, observation):
        return {"action": self._chunk}

    def reset(self, reset_info=None):
        pass

    def configure(self, params):
        return {"applied": params}

    def close(self):
        if self._raise_on_close:
            raise RuntimeError("socket already gone")
        self.closed = True


class _FakeRunner:
    def __init__(self):
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1


def _img_obs(h=32, w=32):
    return {"image": np.zeros((h, w, 3), dtype=np.uint8)}


def _policy(meta, chunk, *, client=None, runner=None):
    client = client or _FakeClient(meta, chunk)
    return (
        VeraPolicy(
            embodiment="pusht",
            auto_launch_server=False,
            client=client,
            server_runner=runner,
        ),
        client,
    )


class TestActionColumnBinding:
    """The chunk->actuator-name mapping selected by ``_action_column_names``."""

    def test_unbound_emits_action_i_keys_and_warns_once(self, caplog):
        meta = {"action_space": "pos", "context_frames": 1, "gripper_dim_index": -1}
        policy, _ = _policy(meta, [[0.1, 0.2, 0.3]])  # no robot_state_keys, no mapping
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.vera.provider"):
            first = asyncio.run(policy.get_actions(_img_obs(), ""))
            # Drain the queued chunk, then force a second infer + bind.
            asyncio.run(policy.get_actions(_img_obs(), ""))
        assert list(first[0].keys()) == ["action_0", "action_1", "action_2"]
        unbound_warnings = [r for r in caplog.records if "UNRESOLVED" in r.getMessage()]
        assert len(unbound_warnings) == 1  # warn-once latch, not per-call spam

    def test_joints_bind_directly_and_truncate_to_action_dim(self):
        meta = {"action_space": "joint_position", "context_frames": 1, "gripper_dim_index": -1}
        policy, _ = _policy(meta, [[0.1, 0.2, 0.3]])
        policy.set_robot_state_keys(["shoulder", "elbow", "wrist", "extra"])
        out = asyncio.run(policy.get_actions(_img_obs(), ""))
        assert list(out[0].keys()) == ["shoulder", "elbow", "wrist"]

    def test_trailing_gripper_column_kept_as_action_extra(self):
        meta = {"action_space": "joint_position", "context_frames": 1, "gripper_dim_index": -1}
        policy, _ = _policy(meta, [[0.1, 0.2, 0.3]])
        policy.set_robot_state_keys(["shoulder", "elbow"])  # fewer joints than columns
        out = asyncio.run(policy.get_actions(_img_obs(), ""))
        assert list(out[0].keys()) == ["shoulder", "elbow", "action_2"]


class TestClose:
    """``close`` is fail-soft on the client and always stops a managed runner."""

    def test_close_swallows_client_error_and_stops_runner(self):
        runner = _FakeRunner()
        client = _FakeClient({"action_space": "pos"}, [[0.0]], raise_on_close=True)
        policy, _ = _policy({}, [[0.0]], client=client, runner=runner)
        policy.close()  # must not raise despite client.close() error
        assert runner.stop_calls == 1

    def test_close_without_runner_is_noop_safe(self):
        client = _FakeClient({}, [[0.0]])
        policy, _ = _policy({}, [[0.0]], client=client, runner=None)
        policy.close()
        assert client.closed is True
