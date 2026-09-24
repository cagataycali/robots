"""Async-RTC overlap in the ``success_fn`` path of :meth:`PolicyRunner.evaluate`.

``run_policy`` overlaps chunk-N+1 inference with chunk-N execution
(:mod:`tests.simulation.test_policy_runner_async_rtc`), but the success-rate
eval path used to run policies fully synchronously with no way to ask for the
overlap - so a chunk-emitting VLA could not be evaluated under the realistic
inference latency it faces in deployment, and the eval payload exposed no
inference-cost telemetry.

``evaluate(..., async_rtc=True)`` (and the ``eval_policy`` facade) now opt into
the same single-worker prefetch pipeline. These tests pin:

* default eval stays synchronous (every inference starts on a chunk boundary)
* opt-in async eval overlaps (an inference STARTS mid-chunk)
* latency is hidden behind execution: an action is dispatched while a
  prefetched query is still in flight, which the synchronous loop cannot do
* both paths report RTC inference telemetry in the payload
* identical step accounting for an observation-independent policy
* the spec/benchmark path rejects ``async_rtc=True`` (reproducible by design)
"""

from __future__ import annotations

import inspect
import threading
import time
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.base import Policy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner

_CHUNK = 4
_INFER_SLEEP = 0.05
_EXEC_SLEEP = 0.02  # per send_action; chunk exec (~0.08s) > infer (0.05s) -> hidden
# See tests.simulation.test_policy_runner_async_rtc._OVERLAP_WAIT.
_OVERLAP_WAIT = 0.25


class _CountingSim(SimEngine):
    """Minimal ``SimEngine`` that counts ``send_action`` calls and can pace them."""

    def __init__(self, exec_sleep: float = 0.0) -> None:
        self._joint_names = ["j0", "j1", "j2"]
        self._robots = {"arm": self._joint_names}
        self._exec_sleep = exec_sleep
        self._lock = threading.Lock()
        self.send_count = 0
        # Latency-masking handshake (see ``_ChunkPolicy`` and
        # ``test_inference_overlaps_execution_only_when_async``): the policy sets
        # ``inference_active`` while a query is open, and every action dispatched
        # in that window sets ``overlap_seen``. Only one query is ever in flight
        # (single prefetch worker), so the pair needs no generation counter.
        self.inference_active = threading.Event()
        self.overlap_seen = threading.Event()

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        with self._lock:
            self.send_count = 0
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
        if self.inference_active.is_set():
            self.overlap_seen.set()
        with self._lock:
            self.send_count += 1
        if self._exec_sleep:
            time.sleep(self._exec_sleep)

    def render(self, camera_name="default", width=None, height=None):
        return {"image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8)}


class _ChunkPolicy(Policy):
    """Emits fixed-size chunks; records the sim step index at each inference start."""

    def __init__(
        self,
        sim: _CountingSim,
        chunk: int = _CHUNK,
        infer_sleep: float = _INFER_SLEEP,
        overlap_wait: float | None = None,
    ) -> None:
        self._sim = sim
        self.actions_per_step = chunk
        self._chunk = chunk
        self._infer_sleep = infer_sleep
        self._overlap_wait = overlap_wait
        self.overlaps: list[bool] = []
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
        if self._overlap_wait is not None:
            # Hold the query open until the loop dispatches another action
            # instead of sleeping a fixed period and timing the run: the overlap
            # is then observed. The synchronous loop is inside this call and
            # cannot dispatch, so it falls through on the timeout.
            self._sim.overlap_seen.clear()
            self._sim.inference_active.set()
            try:
                self.overlaps.append(self._sim.overlap_seen.wait(self._overlap_wait))
            finally:
                self._sim.inference_active.clear()
        elif self._infer_sleep:
            time.sleep(self._infer_sleep)
        keys = self.robot_state_keys or ["j0", "j1", "j2"]
        return [{k: 0.0 for k in keys} for _ in range(self._chunk)]


def _evaluate(
    async_rtc: bool,
    *,
    exec_sleep: float = _EXEC_SLEEP,
    max_steps: int = 16,
    overlap_wait: float | None = None,
) -> tuple[dict, _ChunkPolicy, _CountingSim]:
    sim = _CountingSim(exec_sleep=exec_sleep)
    policy = _ChunkPolicy(sim, overlap_wait=overlap_wait)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    result = PolicyRunner(sim).evaluate(
        "arm",
        policy,
        n_episodes=1,
        max_steps=max_steps,
        action_horizon=_CHUNK,
        control_frequency=50.0,
        async_rtc=async_rtc,
    )
    return result, policy, sim


def _payload(result: dict) -> dict[str, Any]:
    return result["content"][1]["json"]


def test_eval_async_rtc_overlaps_inference_mid_chunk() -> None:
    """The prefetched chunk-N+1 inference fires while chunk N is still draining."""
    result, policy, _ = _evaluate(async_rtc=True)
    assert result["status"] == "success"
    assert len(policy.infer_starts) >= 2
    assert any(c % _CHUNK != 0 for c in policy.infer_starts), policy.infer_starts
    # The first prefetch starts strictly before chunk 1 drains.
    assert policy.infer_starts[1] < _CHUNK, policy.infer_starts


def test_eval_default_is_synchronous() -> None:
    """The default eval never overlaps: every inference is on a chunk boundary."""
    result, policy, _ = _evaluate(async_rtc=False)
    assert result["status"] == "success"
    assert len(policy.infer_starts) >= 2
    assert all(c % _CHUNK == 0 for c in policy.infer_starts), policy.infer_starts


@pytest.mark.parametrize(("async_rtc", "expect_overlap"), [(False, False), (True, True)])
def test_eval_inference_overlaps_execution_only_when_async(async_rtc: bool, expect_overlap: bool) -> None:
    """Latency masking, observed directly instead of inferred from elapsed time.

    Each query is held open until the eval loop dispatches another action. The
    async pipeline drains the rest of the current chunk while the prefetched
    query is in flight, so the overlap is seen; the synchronous loop is inside
    the call and cannot dispatch, so every query falls through on the timeout.
    """
    result, policy, _ = _evaluate(async_rtc=async_rtc, overlap_wait=_OVERLAP_WAIT)
    assert result["status"] == "success"
    assert policy.overlaps, "policy was never queried"
    assert any(policy.overlaps) == expect_overlap, policy.overlaps


def test_eval_async_rtc_reports_telemetry() -> None:
    """Async eval payload proves overlap: a prefetch HIT + inference timing.

    With chunk execution (per-step ``exec_sleep``) longer than inference, the
    prefetched chunk finishes before the seam, so the swap is a hit (inference
    fully hidden) rather than a starvation-block.
    """
    result, _, _ = _evaluate(async_rtc=True, exec_sleep=0.05)
    payload = _payload(result)
    assert payload["rtc_async_enabled"] is True
    assert payload["rtc_chunks_acquired"] >= 2
    assert payload["rtc_prefetch_hits"] >= 1
    assert payload["rtc_avg_inference_ms"] > 0.0


def test_eval_sync_reports_inference_telemetry() -> None:
    """Even synchronous eval reports inference cost (rtc_async_enabled=False)."""
    result, _, _ = _evaluate(async_rtc=False)
    payload = _payload(result)
    assert payload["rtc_async_enabled"] is False
    assert payload["rtc_prefetch_hits"] == 0
    assert payload["rtc_prefetch_blocks"] == 0
    assert payload["rtc_avg_inference_ms"] > 0.0


def test_eval_async_matches_sync_step_accounting() -> None:
    """Both paths run the same number of steps for an obs-independent policy."""
    sync_result, _, _ = _evaluate(async_rtc=False, exec_sleep=0.0)
    async_result, _, _ = _evaluate(async_rtc=True, exec_sleep=0.0)
    sync_json = _payload(sync_result)
    async_json = _payload(async_result)
    assert sync_json["episodes"][0]["steps"] == async_json["episodes"][0]["steps"] == 16
    assert sync_json["success_rate"] == async_json["success_rate"]


def test_eval_async_rtc_rejected_on_spec_path() -> None:
    """The spec/benchmark path stays synchronous: async_rtc=True is rejected."""
    sim = _CountingSim()
    policy = _ChunkPolicy(sim)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    # The guard fires before any spec method is touched, so any non-None spec
    # exercises it.
    result = PolicyRunner(sim).evaluate("arm", policy, spec=object(), async_rtc=True)  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "async_rtc" in result["content"][0]["text"]


def test_eval_async_rtc_defaults_to_false() -> None:
    """Eval defaults to synchronous (reproducible) - async is explicit opt-in."""
    assert inspect.signature(PolicyRunner.evaluate).parameters["async_rtc"].default is False
    assert inspect.signature(SimEngine.eval_policy).parameters["async_rtc"].default is False
