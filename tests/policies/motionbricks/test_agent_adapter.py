"""Unit tests for ``_MotionBricksAgentAdapter`` - the torch marshalling seam.

The adapter wraps the upstream ``full_navigation_agent``. Its :meth:`build`
classmethod needs the ``motionbricks`` package plus the git-LFS checkpoints (and
is covered by ``tests_integ/policies/motionbricks/``), but the per-step
marshalling contract is pure and pinned here against a fake agent: the policy
hands the adapter a plain ``control_signals`` dict and expects it to be turned
into typed torch tensors with the exact dtypes/shapes the generator consumes,
and the returned ``qpos`` to be a ``float64`` ndarray. A silent drift in tensor
dtype/shape (e.g. ``mode`` no longer ``long`` or the token mask losing its
``[1, -1]`` view) would corrupt G1 motion synthesis without raising, so these
tests run on any machine - no GPU, no checkpoints, no ``motionbricks`` install.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from strands_robots.policies.motionbricks.policy import _MotionBricksAgentAdapter


class _FakeFullAgent:
    """Minimal stand-in for the upstream ``full_navigation_agent``.

    Records the control dict + dt forwarded to :meth:`generate_new_frames` so
    the marshalling can be asserted, and returns a fixed frame / context.
    """

    def __init__(self, qpos: list[float]) -> None:
        self._qpos = qpos
        self.context = torch.zeros(1, 4)
        self.reset_calls = 0
        self.generate_calls: list[tuple[dict[str, Any], float]] = []

    def get_next_frame(self) -> list[float]:
        return self._qpos

    def get_context_mujoco_qpos(self) -> torch.Tensor:
        return self.context

    def generate_new_frames(self, control_signals: dict[str, Any], controller_dt: float) -> None:
        self.generate_calls.append((control_signals, controller_dt))

    def reset(self) -> None:
        self.reset_calls += 1


def _make_adapter(fake: _FakeFullAgent) -> _MotionBricksAgentAdapter:
    return _MotionBricksAgentAdapter(
        full_agent=fake,
        clip_keys=["idle", "walk"],
        clip_token_specs=[None, [1, 1, 1]],
        min_token=2,
        max_token=11,
        device="cpu",
    )


def test_init_stores_fields_and_casts_tokens() -> None:
    fake = _FakeFullAgent([0.0])
    adapter = _MotionBricksAgentAdapter(
        full_agent=fake,
        clip_keys=["idle", "walk"],
        clip_token_specs=[None, [1, 1, 1]],
        min_token=2,
        max_token=11,
        device="cpu",
    )

    assert adapter.clip_keys == ["idle", "walk"]
    assert adapter.clip_token_specs == [None, [1, 1, 1]]
    assert adapter.min_token == 2
    assert adapter.max_token == 11
    assert isinstance(adapter.min_token, int)
    assert isinstance(adapter.max_token, int)


def test_reset_delegates_to_agent() -> None:
    fake = _FakeFullAgent([0.0])
    adapter = _make_adapter(fake)

    adapter.reset()
    adapter.reset()

    assert fake.reset_calls == 2


def test_next_qpos_returns_current_frame_as_float64_array() -> None:
    qpos = [0.1, -0.2, 0.3, 1.5]
    fake = _FakeFullAgent(qpos)
    adapter = _make_adapter(fake)

    out = adapter.next_qpos(
        control_signals={
            "movement_direction": [1.0, 0.0, 0.0],
            "facing_direction": [0.0, 1.0, 0.0],
            "mode": 1,
            "allowed_pred_num_tokens": [1, 1, 1, 0],
        },
        controller_dt=0.05,
    )

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, qpos)


def test_next_qpos_marshals_control_signals_to_typed_tensors() -> None:
    fake = _FakeFullAgent([0.0, 1.0])
    adapter = _make_adapter(fake)

    adapter.next_qpos(
        control_signals={
            "movement_direction": [1.0, 0.0, 0.0],
            "facing_direction": [0.0, 1.0, 0.0],
            "mode": 1,
            "allowed_pred_num_tokens": [1, 1, 1, 0],
        },
        controller_dt=0.05,
    )

    assert len(fake.generate_calls) == 1
    torch_cs, dt = fake.generate_calls[0]
    assert dt == 0.05

    move = torch_cs["movement_direction"]
    assert move.dtype == torch.float32
    assert list(move.shape) == [1, 3]
    np.testing.assert_allclose(move.numpy(), [[1.0, 0.0, 0.0]])

    face = torch_cs["facing_direction"]
    assert face.dtype == torch.float32
    assert list(face.shape) == [1, 3]
    np.testing.assert_allclose(face.numpy(), [[0.0, 1.0, 0.0]])

    mode = torch_cs["mode"]
    assert mode.dtype == torch.long
    assert list(mode.shape) == [1, 1]
    assert int(mode.item()) == 1

    tokens = torch_cs["allowed_pred_num_tokens"]
    assert tokens.dtype == torch.int32
    assert list(tokens.shape) == [1, 4]
    assert tokens.flatten().tolist() == [1, 1, 1, 0]

    # The motion context is forwarded verbatim (no copy / reshape).
    assert torch_cs["context_mujoco_qpos"] is fake.context
