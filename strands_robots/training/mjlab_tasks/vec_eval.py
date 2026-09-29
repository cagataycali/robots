"""Vectorized evaluation of ``Policy`` providers on the mjlab backend.

``run_policy`` drives ONE world: observe, ``policy.get_actions``, ``send_action``.
On ``backend="mjlab"`` with ``num_envs=N`` the same loop can drive all N
worlds in lockstep: :meth:`MjlabEngine.get_observation_batch` -> N per-world
observation dicts -> the policy (one batched forward pass when the provider
exposes ``get_actions_batch`` like ``rsl_rl_onnx``, otherwise N concurrent
``get_actions`` calls) -> :meth:`MjlabEngine.send_action_batch`. Each world is
one episode; per-world domain randomization (:meth:`MjlabEngine.randomize`)
and per-world policy kwargs (e.g. a different ``target_pose`` per world) make
the N episodes independent samples.

:class:`BatchedLeRobotRecorder` buffers the N episodes in memory during the
lockstep rollout and flushes them as N LeRobot v3 episodes into one dataset
through the classic :class:`~strands_robots.dataset_recorder.DatasetRecorder`
(one ``add_frame`` stream per episode, ``save_episode`` between them), so the
output is byte-for-byte the format ``run_policy`` records - the only difference
is that the physics for all N ran in parallel. Cameras are not recorded here
(the batched worlds have no renderer; proprio-only datasets).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VecRolloutResult:
    """What one lockstep rollout produced."""

    num_envs: int
    ticks: int
    wall_s: float
    policy_s: float
    physics_s: float
    batched_policy: bool
    final_obs: list[dict[str, float]]
    per_world: list[dict[str, Any]] = field(default_factory=list)
    recorded: dict[str, Any] | None = None

    @property
    def episodes_per_minute(self) -> float:
        """Episodes (= worlds) completed per minute of wall clock."""
        return 60.0 * self.num_envs / self.wall_s if self.wall_s > 0 else float("inf")

    def summary(self) -> dict[str, Any]:
        """JSON-friendly throughput summary."""
        out = {
            "num_envs": self.num_envs,
            "ticks": self.ticks,
            "wall_s": round(self.wall_s, 3),
            "policy_s": round(self.policy_s, 3),
            "physics_s": round(self.physics_s, 3),
            "batched_policy": self.batched_policy,
            "episodes_per_minute": round(self.episodes_per_minute, 2),
            "env_ticks_per_s": round(self.num_envs * self.ticks / self.wall_s, 1) if self.wall_s else None,
        }
        if self.recorded is not None:
            out["recorded"] = self.recorded
        return out


def split_batch(obs_batch: dict[str, Any], num_envs: int) -> list[dict[str, Any]]:
    """``{key: (N, ...)}`` tensors -> N plain-float observation dicts (what ``Policy.get_actions`` eats)."""
    host: dict[str, np.ndarray] = {}
    for k, v in obs_batch.items():
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        host[k] = arr
    out: list[dict[str, Any]] = []
    for i in range(num_envs):
        d: dict[str, Any] = {}
        for k, arr in host.items():
            row = arr[i]
            d[k] = float(row) if np.ndim(row) == 0 else row.astype(np.float32).tolist()
        out.append(d)
    return out


class BatchedLeRobotRecorder:
    """N in-memory episode buffers, flushed as N episodes of one LeRobot v3 dataset.

    ``add_tick(observations, actions)`` appends one frame to every world's buffer;
    ``flush(recorder)`` replays world 0..N-1 through ``DatasetRecorder.add_frame``
    + ``save_episode``. Worlds whose ``done`` flag was set keep only the frames up
    to that tick.
    """

    def __init__(self, num_envs: int, joint_names: Sequence[str], action_keys: Sequence[str], task: str) -> None:
        self.num_envs = int(num_envs)
        self.joint_names = list(joint_names)
        self.action_keys = list(action_keys)
        self.task = task
        self._frames: list[list[tuple[dict[str, Any], dict[str, Any]]]] = [[] for _ in range(self.num_envs)]
        self._done = np.zeros(self.num_envs, dtype=bool)

    def add_tick(self, observations: Sequence[dict[str, Any]], actions: Sequence[dict[str, Any]]) -> None:
        """Append one frame per world (skipping worlds already marked done)."""
        for i, (o, a) in enumerate(zip(observations, actions, strict=True)):
            if self._done[i]:
                continue
            scalars = {k: v for k, v in o.items() if np.ndim(v) == 0 or k.startswith("base_")}
            self._frames[i].append((scalars, dict(a)))

    def mark_done(self, world: int) -> None:
        """Stop buffering frames for ``world`` (its episode ended early)."""
        self._done[world] = True

    @property
    def frames_buffered(self) -> int:
        """Total frames held in memory across worlds."""
        return sum(len(f) for f in self._frames)

    def flush(self, recorder: Any) -> dict[str, Any]:
        """Write every buffered world as one episode; returns counts + timings."""
        t0 = time.perf_counter()
        episodes = 0
        frames = 0
        for i, world_frames in enumerate(self._frames):
            if not world_frames:
                continue
            for obs, act in world_frames:
                recorder.add_frame(observation=obs, action=act, task=self.task, required_action_keys=self.action_keys)
                frames += 1
            recorder.save_episode()
            episodes += 1
            logger.debug("flushed world %d: %d frames", i, len(world_frames))
        return {"episodes": episodes, "frames": frames, "flush_s": round(time.perf_counter() - t0, 3)}


def open_recorder(engine: Any, repo_id: str, root: str | Path, fps: int, task: str) -> Any:
    """A classic ``DatasetRecorder`` whose schema is the engine's live scene (proprio only)."""
    from strands_robots.dataset_recorder import DatasetRecorder

    joint_names, action_names, _cams, _dims, robot_type, _rec_cams, base_specs = engine._collect_recording_schema()
    return DatasetRecorder.create(
        repo_id=repo_id,
        fps=int(fps),
        robot_type=robot_type,
        camera_keys=[],
        camera_dims={},
        joint_names=joint_names,
        action_names=action_names,
        root=str(root),
        use_videos=False,
        task=task,
        extra_state_specs=base_specs or None,
    )


async def vec_rollout(
    engine: Any,
    policy: Any,
    *,
    robot_name: str,
    ticks: int,
    instruction: str = "",
    control_hz: float = 50.0,
    kwargs_per_world: Sequence[dict[str, Any]] | None = None,
    recorder: BatchedLeRobotRecorder | None = None,
    on_tick: Callable[[int, list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
    force_unbatched: bool = False,
) -> VecRolloutResult:
    """Drive all ``engine.num_envs`` worlds for ``ticks`` control steps with one policy.

    Args:
        engine: A built ``MjlabEngine`` (any ``num_envs``); ``reset()`` is called first.
        policy: A ``Policy`` provider. ``get_actions_batch(observations, instruction,
            kwargs_per_world=...)`` is used when present, else N concurrent ``get_actions``.
        robot_name: Which robot to drive.
        ticks: Control steps per episode (all worlds run the same length).
        control_hz: Policy rate; physics substeps = round(1 / (dt * control_hz)).
        kwargs_per_world: Per-world ``get_actions`` kwargs (e.g. ``target_pose``).
        recorder: Optional :class:`BatchedLeRobotRecorder` fed every tick.
        on_tick: Optional callback ``(tick, observations, actions)`` for metrics.
        force_unbatched: Ignore ``get_actions_batch`` (for the throughput comparison).
    """
    n = int(engine.num_envs)
    kw = list(kwargs_per_world) if kwargs_per_world is not None else [{}] * n
    if len(kw) != n:
        raise ValueError(f"kwargs_per_world has {len(kw)} entries for {n} worlds")
    engine.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    action_keys = list(engine.robot_action_keys(robot_name))
    dt = float(engine.physics_timestep() or 0.002)
    n_sub = max(1, int(round(1.0 / (dt * control_hz))))
    batched = hasattr(policy, "get_actions_batch") and not force_unbatched

    t_start = time.perf_counter()
    policy_s = 0.0
    physics_s = 0.0
    observations: list[dict[str, Any]] = []
    for tick in range(int(ticks)):
        t0 = time.perf_counter()
        observations = split_batch(engine.get_observation_batch(robot_name), n)
        if batched:
            actions = await policy.get_actions_batch(observations, instruction, kwargs_per_world=kw)
        else:
            chunks = await asyncio.gather(
                *(policy.get_actions(o, instruction, **k) for o, k in zip(observations, kw, strict=True))
            )
            actions = [c[0] for c in chunks]
        t1 = time.perf_counter()
        block = np.asarray([[float(a.get(k, 0.0)) for k in action_keys] for a in actions], dtype=np.float32)
        res = engine.send_action_batch(block, robot_name, n_substeps=n_sub)
        if res.get("status") != "success":
            raise RuntimeError(res["content"][0]["text"])
        t2 = time.perf_counter()
        policy_s += t1 - t0
        physics_s += t2 - t1
        if recorder is not None:
            recorder.add_tick(observations, actions)
        if on_tick is not None:
            on_tick(tick, observations, actions)
    final = split_batch(engine.get_observation_batch(robot_name), n)
    return VecRolloutResult(
        num_envs=n,
        ticks=int(ticks),
        wall_s=time.perf_counter() - t_start,
        policy_s=policy_s,
        physics_s=physics_s,
        batched_policy=batched,
        final_obs=[{k: v for k, v in f.items() if np.ndim(v) == 0} for f in final],
    )
