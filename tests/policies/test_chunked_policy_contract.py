"""Contract tests for the ``ChunkedPolicy`` introspection protocol and the
shared ``resolve_chunk_length`` chunk-sizing rule.

A chunk-emitting policy (ACT, diffusion, pi0, SmolVLA, MolmoAct2) returns N
actions per ``get_actions`` call and expects all N consumed open-loop before a
re-query. Every consumer - the single-policy runner, the multi-episode eval
loop, and the synchronized multi-robot loop - must size that chunk identically.
These tests pin the typed contract and the one helper they all route through,
independent of any real model weights, so the chunk-sizing rule cannot drift
back to per-consumer copies.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.policies import ChunkedPolicy, MockPolicy, resolve_chunk_length
from strands_robots.policies.base import Policy


class MockChunkedPolicy(Policy):
    """Weight-free chunk-emitting policy for driver/contract tests.

    Returns a fixed-length chunk of zero actions and declares the
    ``ChunkedPolicy`` contract attributes, so tests can assert chunk-consumption
    behaviour without loading a real VLA checkpoint.
    """

    requires_images = False

    def __init__(self, actions_per_step: int = 10, supports_rtc: bool = False) -> None:
        self.actions_per_step = actions_per_step
        self.supports_rtc = supports_rtc
        self.calls = 0
        self._keys: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock_chunked"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = list(robot_state_keys)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls += 1
        keys = self._keys or ["j0", "j1", "j2"]
        return [{k: 0.0 for k in keys} for _ in range(self.actions_per_step)]


def test_mock_chunked_policy_satisfies_protocol() -> None:
    """A policy declaring the contract attrs is recognised at runtime."""
    assert isinstance(MockChunkedPolicy(), ChunkedPolicy)


def test_single_action_policy_is_not_chunked() -> None:
    """A single-action provider does not satisfy the chunked contract."""
    assert not isinstance(MockPolicy(), ChunkedPolicy)


def test_protocol_requires_both_attributes() -> None:
    """Declaring only one of the two contract attrs is not enough."""

    class OnlyChunk:
        actions_per_step = 8

    class OnlyRtc:
        supports_rtc = True

    assert not isinstance(OnlyChunk(), ChunkedPolicy)
    assert not isinstance(OnlyRtc(), ChunkedPolicy)


def test_resolve_chunk_length_honors_actions_per_step() -> None:
    """A chunk longer than the horizon is kept whole (no tail dropped)."""
    pol = MockChunkedPolicy(actions_per_step=30)
    assert resolve_chunk_length(pol, action_horizon=8) == 30


def test_resolve_chunk_length_uses_horizon_when_larger() -> None:
    """A horizon larger than the policy's chunk wins."""
    pol = MockChunkedPolicy(actions_per_step=4)
    assert resolve_chunk_length(pol, action_horizon=16) == 16


def test_resolve_chunk_length_defaults_single_action_policy() -> None:
    """A policy without ``actions_per_step`` is treated as a 1-action chunk."""
    assert resolve_chunk_length(MockPolicy(), action_horizon=8) == 8
    assert resolve_chunk_length(MockPolicy(), action_horizon=1) == 1


def test_resolve_chunk_length_clamps_nonpositive_inputs() -> None:
    """Degenerate horizon / chunk values clamp to at least one action."""
    assert resolve_chunk_length(MockChunkedPolicy(actions_per_step=0), action_horizon=0) == 1
    assert resolve_chunk_length(MockChunkedPolicy(actions_per_step=-5), action_horizon=0) == 1


def test_resolve_chunk_length_ignores_non_integer_attr() -> None:
    """A non-integer ``actions_per_step`` falls back to single-action.

    Guards the defensive coercion branch: a malformed attribute (here a str)
    must not blow up the control loop, it degrades to a 1-action chunk. The
    ``type: ignore`` is intentional - we deliberately feed a duck type that
    violates the ``Policy`` signature to exercise that runtime defence.
    """

    class Bad:
        actions_per_step = "lots"

    assert resolve_chunk_length(Bad(), action_horizon=5) == 5  # type: ignore[arg-type]


def test_chunked_policy_accepts_property_backed_attrs() -> None:
    """The contract is satisfied by property-backed attributes too.

    ``LerobotLocalPolicy`` exposes ``supports_rtc`` as a property over its
    internal RTC state; the protocol must recognise that shape.
    """

    class PropBacked:
        actions_per_step = 16

        @property
        def supports_rtc(self) -> bool:
            return True

    assert isinstance(PropBacked(), ChunkedPolicy)


@pytest.mark.parametrize("horizon,chunk,expected", [(8, 50, 50), (8, 1, 8), (1, 1, 1), (30, 30, 30)])
def test_resolve_chunk_length_matrix(horizon: int, chunk: int, expected: int) -> None:
    assert resolve_chunk_length(MockChunkedPolicy(actions_per_step=chunk), horizon) == expected
