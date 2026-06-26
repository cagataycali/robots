"""Async-RTC chunk pipeline in :class:`PolicyRunner.run`.

The synchronous loop queries the policy, fully drains the returned chunk, then
re-queries - so inference never overlaps execution and a chunk-emitting VLA
shows a per-seam stall in sim that it would NOT show on real hardware (where an
async controller hides inference latency behind chunk execution).

``run(..., async_rtc=True)`` overlaps the two: while the current chunk drains,
the next ``get_actions`` is fired on a single background worker once the chunk
is ~50% consumed, then atomically swapped in. These tests pin:

* overlap: the next inference STARTS mid-chunk (before the current chunk drains)
* synchronous baseline: inference only starts after the chunk fully drains
* latency masking: async wall-time is materially lower than synchronous
* correctness/back-compat: identical step accounting; default is synchronous
"""

from __future__ import annotations

import inspect
import threading
import time
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner

_CHUNK = 4
_INFER_SLEEP = 0.05
_EXEC_SLEEP = 0.02  # per send_action; chunk exec (~0.08s) > infer (0.05s) -> hidden


class _CountingSim(SimEngine):
    """Minimal ``SimEngine`` that counts ``send_action`` calls and can pace them.

    ``send_count`` is incremented BEFORE the optional per-action sleep so a
    concurrently-running policy that reads it observes the step index reached so
    far, which is what the overlap assertions key off.
    """

    def __init__(self, exec_sleep: float = 0.0) -> None:
        self._joint_names = ["j0", "j1", "j2"]
        self._robots = {"arm": self._joint_names}
        self._exec_sleep = exec_sleep
        self._lock = threading.Lock()
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
        return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._robots.get(robot_name, []))

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {n: 0.0 for n in self._joint_names}

    def send_action(self, action, robot_name=None, n_substeps=1):
        with self._lock:
            self.send_count += 1
        if self._exec_sleep:
            time.sleep(self._exec_sleep)

    def render(self, camera_name="default", width=None, height=None):
        return {"image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8)}


class _ChunkPolicy(Policy):
    """Emits fixed-size chunks; records the sim step index at each inference start.

    ``infer_starts[k]`` is the value of ``sim.send_count`` captured at the very
    first line of the k-th ``get_actions`` call (before the inference sleep). In
    the synchronous loop these are exact multiples of the chunk size (the chunk
    is always fully drained before the next query); the async pipeline fires the
    query mid-chunk, so at least one value is NOT a chunk multiple.
    """

    def __init__(self, sim: _CountingSim, chunk: int = _CHUNK, infer_sleep: float = _INFER_SLEEP) -> None:
        self._sim = sim
        self.actions_per_step = chunk
        self._chunk = chunk
        self._infer_sleep = infer_sleep
        self.robot_state_keys: list[str] = []
        self.infer_starts: list[int] = []
        self._lock = threading.Lock()

    @property
    def provider_name(self) -> str:
        return "chunk-test"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = robot_state_keys

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        with self._lock:
            self.infer_starts.append(self._sim.send_count)
        if self._infer_sleep:
            time.sleep(self._infer_sleep)
        keys = self.robot_state_keys or ["j0", "j1", "j2"]
        return [{k: 0.0 for k in keys} for _ in range(self._chunk)]


def _run(
    async_rtc: bool, *, exec_sleep: float = _EXEC_SLEEP, n_steps: int = 16
) -> tuple[dict, _ChunkPolicy, _CountingSim]:
    sim = _CountingSim(exec_sleep=exec_sleep)
    policy = _ChunkPolicy(sim)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    duration = n_steps / 50.0
    result = PolicyRunner(sim).run(
        "arm",
        policy,
        duration=duration,
        control_frequency=50.0,
        action_horizon=_CHUNK,
        fast_mode=True,
        async_rtc=async_rtc,
    )
    return result, policy, sim


def test_async_rtc_starts_next_inference_mid_chunk() -> None:
    """The prefetched chunk-N+1 inference fires while chunk N is still draining."""
    result, policy, sim = _run(async_rtc=True)
    assert result["status"] == "success"
    # More than one chunk was consumed, so prefetch had to fire.
    assert len(policy.infer_starts) >= 2
    # At least one inference began at a non-chunk-boundary step index -> the
    # query overlapped execution rather than waiting for a full drain.
    assert any(c % _CHUNK != 0 for c in policy.infer_starts), policy.infer_starts
    # Specifically the first prefetch starts strictly before chunk 1 drains.
    assert policy.infer_starts[1] < _CHUNK, policy.infer_starts


def test_sync_only_queries_after_chunk_drains() -> None:
    """The synchronous baseline never overlaps: every query is on a boundary."""
    result, policy, sim = _run(async_rtc=False)
    assert result["status"] == "success"
    assert len(policy.infer_starts) >= 2
    assert all(c % _CHUNK == 0 for c in policy.infer_starts), policy.infer_starts


def test_async_rtc_masks_inference_latency() -> None:
    """Async wall-time is materially lower because inference hides behind exec."""
    t0 = time.perf_counter()
    sync_result, _, _ = _run(async_rtc=False)
    sync_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    async_result, _, _ = _run(async_rtc=True)
    async_elapsed = time.perf_counter() - t0

    assert sync_result["status"] == "success"
    assert async_result["status"] == "success"
    # 16 steps / chunk 4 => 4 chunks. Sync pays infer_sleep per chunk serially;
    # async hides all but (at most) the first. Saving >= 2 inference periods is
    # a conservative, non-flaky margin.
    assert sync_elapsed - async_elapsed > 2 * _INFER_SLEEP, (sync_elapsed, async_elapsed)


def test_async_rtc_matches_sync_step_accounting() -> None:
    """Both paths run the same number of steps and send the same action count."""
    sync_result, _, sync_sim = _run(async_rtc=False, exec_sleep=0.0)
    async_result, _, async_sim = _run(async_rtc=True, exec_sleep=0.0)

    sync_json = sync_result["content"][1]["json"]
    async_json = async_result["content"][1]["json"]
    assert sync_json["n_steps"] == async_json["n_steps"] == 16
    assert sync_sim.send_count == async_sim.send_count == 16
    assert async_json["action_errors"] == 0


def test_async_rtc_defaults_to_false() -> None:
    """Back-compat: async_rtc is opt-in on both PolicyRunner.run and run_policy."""
    assert inspect.signature(PolicyRunner.run).parameters["async_rtc"].default is False
    assert inspect.signature(SimEngine.run_policy).parameters["async_rtc"].default is False
