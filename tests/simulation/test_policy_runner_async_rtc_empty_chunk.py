"""Empty-chunk contract of the async-RTC loop in :meth:`PolicyRunner.run`.

``run(..., async_rtc=True)`` overlaps inference for chunk N+1 with the drain of
chunk N. The corresponding synchronous ``_ChunkPipeline`` seam already pins how
an empty action chunk is handled (see ``test_chunk_pipeline_contract``), but the
non-recording ``PolicyRunner.run`` path carries its own inline copy of that
loop. These tests pin the same "no silent no-op" contract on that inline path so
the two implementations cannot drift:

* an empty FIRST chunk aborts the rollout loudly (never a silent zero-step run);
* a single transient empty chunk mid-rollout degrades to ONE synchronous
  re-query and recovers if that re-query is non-empty (a healthy rollout is not
  killed by one hiccup);
* an empty chunk on BOTH the prefetch and the synchronous re-query aborts loudly
  rather than spinning forever.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from strands_robots.policies.base import Policy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner

_JOINTS = ["j0", "j1", "j2"]
_CHUNK = 2


class _Sim(SimEngine):
    """Minimal no-physics ``SimEngine`` sufficient to drive ``PolicyRunner``."""

    def __init__(self) -> None:
        self._robots = {"arm": _JOINTS}
        self.send_count = 0

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        return {"status": "success"}

    def get_state(self):
        return {"sim_time": 0.0, "step_count": self.send_count}

    def add_robot(self, name, **kw):
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return list(self._robots)

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._robots.get(robot_name, []))

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {n: 0.0 for n in _JOINTS}

    def send_action(self, action, robot_name=None, n_substeps=1):
        self.send_count += 1

    def render(self, camera_name="default", width=None, height=None):
        return {"image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8)}


class _ScriptedChunkPolicy(Policy):
    """Emits a scripted sequence of chunk sizes; 0 means "empty chunk".

    The last size is held for every query past the end of the script, so a
    trailing 0 models a policy that has permanently gone empty.
    """

    def __init__(self, sizes: list[int]) -> None:
        self._sizes = sizes
        self._calls = 0
        self.actions_per_step = _CHUNK
        self.robot_state_keys: list[str] = []

    @property
    def provider_name(self) -> str:
        return "scripted-chunk"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = robot_state_keys

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        idx = min(self._calls, len(self._sizes) - 1)
        self._calls += 1
        n = self._sizes[idx]
        keys = self.robot_state_keys or _JOINTS
        return [{k: 0.0 for k in keys} for _ in range(n)]


def _run(sizes: list[int], *, n_steps: int = 12):
    sim = _Sim()
    policy = _ScriptedChunkPolicy(sizes)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    return sim, PolicyRunner(sim).run(
        "arm",
        policy,
        duration=n_steps / 50.0,
        control_frequency=50.0,
        action_horizon=_CHUNK,
        fast_mode=True,
        async_rtc=True,
    )


def test_async_rtc_initial_empty_chunk_surfaces_error() -> None:
    """An empty FIRST chunk aborts the rollout loudly, never a silent no-op.

    ``run`` surfaces the failure as a ``status=error`` result (the same facade
    contract the sync path uses) rather than reporting a bogus zero-step
    success.
    """
    sim, result = _run([0])
    assert result["status"] == "error", result
    assert "empty action chunk" in result["content"][0]["text"]
    assert sim.send_count == 0  # not one action was applied


def test_async_rtc_empty_chunk_twice_surfaces_error() -> None:
    """Empty on the prefetch AND the synchronous re-query aborts loudly."""
    _sim, result = _run([_CHUNK, 0])
    assert result["status"] == "error", result
    assert "empty action chunk twice" in result["content"][0]["text"]


def test_async_rtc_transient_empty_chunk_recovers() -> None:
    """One empty chunk mid-rollout degrades to a synchronous re-query and recovers.

    A single transient empty result must not kill an otherwise-healthy rollout:
    the loop re-queries once, gets a non-empty chunk, and runs to completion.
    """
    sim, result = _run([_CHUNK, 0, _CHUNK], n_steps=12)
    assert result["status"] == "success", result
    assert sim.send_count > 0
