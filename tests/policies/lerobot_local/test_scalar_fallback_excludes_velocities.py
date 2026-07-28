# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The state fallback must not interleave velocities into the state vector.

When the configured ``robot_state_keys`` match nothing in the observation, the
ordering falls back to the observation's own scalar keys. The MuJoCo backend emits
a ``"<joint>.vel"`` sibling per joint, and the unfiltered comprehension took them
in insertion order - so ``observation.state`` became
``[pos0, vel0, pos1, vel1, ...]``: twice the DOF count, then truncated to the
model's dim, so HALF the slots held velocities and the trailing joints were
dropped entirely.

Measured on a 6-DoF arm driven to distinct known values::

    resolved order       ['1','1.vel','2','2.vel','3','3.vel','4','4.vel','5','5.vel','6','6.vel']
    first 6 (6-dim model) [0.2981, -0.47,  -0.1973, 0.6561, 0.4978, -0.5439]
    CORRECT positions     [0.2981, -0.1973, 0.4978, 0.0987, -0.3974, 0.2474]

That is a wrong state vector with no error. A standalone ``.vel`` key with no
position companion is KEPT, because some embodiments legitimately declare velocity
state (``embodiments.json`` has ``x.vel`` / ``y.vel`` / ``theta.vel`` for mobile
bases) and dropping it would corrupt those instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from strands_robots.policies.lerobot_local.policy import (
    LerobotLocalPolicy,
    _state_fallback_scalar_keys,
)

_POSITIONS = [0.2981, -0.1973, 0.4978, 0.0987, -0.3974, 0.2474]
_VELOCITIES = [-0.47, 0.6561, -0.5439, 0.11, -0.22, 0.33]


def _sim_observation() -> dict:
    """A MuJoCo-shaped observation: a '.vel' sibling per joint, interleaved."""
    obs: dict = {}
    for index, (pos, vel) in enumerate(zip(_POSITIONS, _VELOCITIES, strict=True), start=1):
        obs[str(index)] = pos
        obs[f"{index}.vel"] = vel
    return obs


def _state(dim: int) -> SimpleNamespace:
    return SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(dim,))


class TestTheFilterItself:
    def test_velocity_companions_are_dropped(self):
        assert _state_fallback_scalar_keys(_sim_observation()) == [str(i) for i in range(1, 7)]

    def test_a_standalone_velocity_key_is_kept(self):
        """Mobile-base embodiments declare x.vel/y.vel/theta.vel as real state."""
        obs = {"x.vel": 1.0, "y.vel": 2.0, "theta.vel": 3.0}

        assert _state_fallback_scalar_keys(obs) == ["x.vel", "y.vel", "theta.vel"]

    def test_only_the_companion_is_dropped(self):
        obs = {"a": 1.0, "a.vel": 2.0, "b.vel": 3.0}

        assert _state_fallback_scalar_keys(obs) == ["a", "b.vel"]

    def test_task_and_images_are_still_excluded(self):
        obs = {"1": 1.0, "task": "pick", "front": np.zeros((4, 4, 3))}

        assert _state_fallback_scalar_keys(obs) == ["1"]

    def test_order_is_preserved(self):
        """The fallback is POSITIONAL, so observation order is the contract."""
        obs = {"c": 1.0, "a": 2.0, "b": 3.0}

        assert _state_fallback_scalar_keys(obs) == ["c", "a", "b"]

    def test_a_one_dim_array_scalar_is_kept(self):
        """Only >=2-d arrays are camera frames."""
        obs = {"1": 1.0, "quat": np.zeros(4)}

        assert _state_fallback_scalar_keys(obs) == ["1", "quat"]


class TestTheResolvedOrderInThePolicy:
    def _policy(self, state_dim: int = 6, **kwargs) -> LerobotLocalPolicy:
        policy = LerobotLocalPolicy(**kwargs)
        policy._input_features = {"observation.state": _state(state_dim)}
        policy._device = torch.device("cpu")
        # Generic keys that match NOTHING in a named-joint observation, forcing
        # the fallback that this defect lived in.
        policy.set_robot_state_keys([f"joint_{i}" for i in range(6)])
        return policy

    def test_the_fallback_order_is_positions_only(self):
        """Regression: the order used to alternate position and velocity."""
        policy = self._policy(state_dim=12)  # 12 avoids the exact-multiple raise
        obs = _sim_observation()

        order = policy._resolve_state_order(obs, _state_fallback_scalar_keys(obs))

        assert order == [str(i) for i in range(1, 7)]
        assert not any(k.endswith(".vel") for k in order)

    def test_the_packed_values_are_the_joint_positions(self):
        """The whole point: the numbers reaching the model must be the positions."""
        policy = self._policy(state_dim=12)
        obs = _sim_observation()

        order = policy._resolve_state_order(obs, _state_fallback_scalar_keys(obs))
        values = policy._collect_state_values(obs, order)

        assert values == pytest.approx(_POSITIONS)


class TestExactMultipleRaises:
    def _policy(self, state_dim: int) -> LerobotLocalPolicy:
        policy = LerobotLocalPolicy()
        policy._input_features = {"observation.state": _state(state_dim)}
        policy._device = torch.device("cpu")
        policy.set_robot_state_keys([f"joint_{i}" for i in range(state_dim)])
        return policy

    def test_an_exact_multiple_of_the_model_dim_raises(self):
        """N channels per joint would interleave then truncate: half the slots wrong.

        Raises regardless of strict_keys, because it is wrong motion rather than a
        degraded binding.
        """
        policy = self._policy(state_dim=6)
        # 12 scalars, none a '.vel' companion, so the filter cannot save it.
        obs = {f"ch{i}": float(i) for i in range(12)}

        with pytest.raises(ValueError) as excinfo:
            policy._resolve_state_order(obs, _state_fallback_scalar_keys(obs))

        message = str(excinfo.value)
        assert "exact multiple" in message
        assert "2 channels per joint" in message
        assert "set_robot_state_keys" in message

    def test_a_non_multiple_still_only_warns(self):
        """A merely-different count is the pre-existing degraded-binding case."""
        policy = self._policy(state_dim=6)
        obs = {f"ch{i}": float(i) for i in range(7)}

        order = policy._resolve_state_order(obs, _state_fallback_scalar_keys(obs))

        assert order == list(obs)
        assert policy.generic_state_keys_used is True

    def test_an_equal_count_does_not_raise(self):
        """len == expected is not pollution, it is the normal fallback."""
        policy = self._policy(state_dim=6)
        obs = {f"ch{i}": float(i) for i in range(6)}

        assert policy._resolve_state_order(obs, _state_fallback_scalar_keys(obs)) == list(obs)

    def test_the_filter_prevents_the_raise_for_the_real_sim_shape(self):
        """A MuJoCo observation is 2x the dim, but the filter fixes it first."""
        policy = self._policy(state_dim=6)
        obs = _sim_observation()  # 12 keys, 6 of them '.vel' companions

        order = policy._resolve_state_order(obs, _state_fallback_scalar_keys(obs))

        assert order == [str(i) for i in range(1, 7)]

    def test_error_message_is_plain_ascii(self):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        policy = self._policy(state_dim=6)
        obs = {f"ch{i}": float(i) for i in range(12)}

        with pytest.raises(ValueError) as excinfo:
            policy._resolve_state_order(obs, _state_fallback_scalar_keys(obs))

        assert str(excinfo.value).isascii()
