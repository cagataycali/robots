"""Tests for ``strands_robots.policies.base.Policy`` ABC contract.

Covers the ``get_actions_sync`` event-loop dispatch paths: the 'no loop'
fast path and the 'already-in-event-loop' ThreadPoolExecutor fallback.
"""

from __future__ import annotations

import asyncio
from typing import Any

from strands_robots.policies.base import Policy


class _IdentityPolicy(Policy):
    """Minimal concrete Policy for testing Policy ABC's sync wrapper."""

    def __init__(self) -> None:
        self._keys = ["j0"]

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"j0": 0.1}, {"j0": 0.2}]

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = list(robot_state_keys)

    @property
    def provider_name(self) -> str:
        return "identity"


def test_get_actions_sync_outside_event_loop_uses_asyncio_run():
    p = _IdentityPolicy()
    actions = p.get_actions_sync({"observation.state": [0.0]}, instruction="hi")
    assert actions == [{"j0": 0.1}, {"j0": 0.2}]


def test_get_actions_sync_inside_event_loop_uses_threadpool():
    """When called from within a running event loop, the sync wrapper must
    off-load to a thread pool instead of raising 'already in a loop'."""
    p = _IdentityPolicy()

    async def inner():
        # Calling the sync wrapper here forces the thread-pool branch
        return p.get_actions_sync({"observation.state": [0.0]}, instruction="hi")

    actions = asyncio.run(inner())
    assert actions == [{"j0": 0.1}, {"j0": 0.2}]


def test_provider_name_and_state_keys():
    p = _IdentityPolicy()
    assert p.provider_name == "identity"
    p.set_robot_state_keys(["a", "b", "c"])
    assert p._keys == ["a", "b", "c"]
