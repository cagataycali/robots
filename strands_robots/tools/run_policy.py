"""Multi-episode policy rollout - Strands Agent ``@tool`` wrapper.

Closes the fabrication vector identified in
`strands-labs/robots#708 <https://github.com/strands-labs/robots/issues/708>`_:

The existing ``Robot``/``Simulation`` AgentTool surface exposes a
**single-rollout** ``run_policy`` action (one ``duration``/``n_steps`` call
=> one trajectory). When a human asks an LLM agent to "run 20 episodes of
60 steps each", the LLM has no ``n_episodes`` knob to turn - it must
improvise. The historical failure mode (audited across 47 molmoact-e2e
runs, 16 falsely marked OK) is that the LLM dispatches **one** giant
``run_policy`` call, then **narrates** "20/20 episodes complete" - the
recorder sees one mega-episode of ``20x60=1200`` frames and writes
``info.json:total_episodes=1``.

PR #716 fixed the *recorder* side (per-episode ``save_episode`` boundaries
are now wired in ``PolicyRunner.evaluate`` / ``_evaluate_with_spec``). This
tool fixes the *exposure* side: it surfaces ``n_episodes`` explicitly and
drives the episode loop in deterministic Python - no LLM in the loop.

The tool also returns **parquet-truth**, not agent self-report: after the
final ``stop_recording`` it reads ``meta/info.json:total_episodes`` from
the dataset on disk and surfaces that count in the returned payload, so a
downstream verifier comparing "requested vs actual" catches any silent
collapse before status=OK is reported.

Design notes (the contract this tool pins):

* ``simulation`` is a **Python handle**, not an LLM-supplied string. Pass
  the live ``Simulation`` (or ``Robot``-compatible engine) constructed by
  the orchestrator. LLMs cannot synthesize this argument, by design - the
  tool is meant to be invoked from a deterministic outer loop in a
  scripted runner (the pattern voted in HB#349). Mesh-clients drive it
  through normal Python wiring.
* ``n_episodes`` is a required, validated integer. There is no fallback,
  no "infer from duration", no per-episode self-report - the loop iterates
  exactly ``n_episodes`` times, and the parquet-truth gate at the end
  catches any divergence.
* The episode loop calls ``simulation.run_policy(...)`` per iteration and
  invokes the ``PolicyRunner._finalize_recorder_episode`` helper (added
  in PR #716) between rollouts so each episode lands in its own parquet
  row. The trailing ``stop_recording`` flushes the final episode and
  closes the dataset.
* Recording is OPTIONAL. When ``dataset_root`` is provided we drive a full
  ``start_recording`` -> ``stop_recording`` cycle and report parquet-truth
  (``total_episodes``, ``total_frames``). When ``dataset_root`` is omitted
  we still run the N-episode loop but skip recording - useful for smoke
  tests where the goal is just to exercise the policy.

See ``strands-labs/robots#708`` for the full root-cause analysis and the
e2e_agent_test.py fix history (HB#352 -> #716 -> this tool).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from strands.tools.decorator import tool

logger = logging.getLogger(__name__)


def _ok(text: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "success", "content": [{"text": text}]}
    out.update(extra)
    return out


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _read_parquet_truth(dataset_root: str | Path) -> dict[str, Any]:
    """Read ground-truth episode/frame counts from ``meta/info.json``.

    ``info.json`` is sync-flushed by LeRobot v3, so it is the authoritative
    source for ``total_episodes`` / ``total_frames`` immediately after
    ``stop_recording`` returns. The episodes/data parquet files are
    async-flushed and can lag (see HB#372 forensics + e2e verifier's
    two-phase wait pattern), so we explicitly DO NOT depend on them here.

    Returns a partial result on missing fields rather than raising, so the
    caller can surface a structured error instead of a stack trace.
    """
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.is_file():
        return {"info_present": False, "info_path": str(info_path)}
    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"info_present": False, "info_path": str(info_path), "error": repr(e)}
    return {
        "info_present": True,
        "info_path": str(info_path),
        "total_episodes": int(info.get("total_episodes", -1)),
        "total_frames": int(info.get("total_frames", -1)),
        "fps": info.get("fps"),
    }


@tool
def run_policy(
    simulation: Any,
    *,
    robot_name: str | None = None,
    policy_provider: str = "mock",
    policy_config: dict[str, Any] | None = None,
    instruction: str = "",
    n_episodes: int = 1,
    n_steps: int = 60,
    control_frequency: float = 30.0,
    action_horizon: int = 8,
    fast_mode: bool = True,
    dataset_root: str | None = None,
    dataset_repo_id: str = "local/run_policy_rollout",
    dataset_task: str = "",
    dataset_fps: int = 30,
    seed: int | None = None,
    policy_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Roll out a policy for ``n_episodes`` x ``n_steps`` with per-episode parquet boundaries.

    Pass-through wrapper around :meth:`Simulation.run_policy` that owns the
    multi-episode loop and the recording lifecycle, so an LLM agent never
    has to improvise either. Closes the #708 fabrication vector by:

    1. **Explicit ``n_episodes``** - the loop iterates exactly N times,
       no narrated counts.
    2. **Per-episode ``save_episode``** - each rollout lands in its own
       parquet row via ``PolicyRunner._finalize_recorder_episode``
       (wired by PR #716).
    3. **Parquet-truth return** - final payload carries
       ``total_episodes`` / ``total_frames`` read from
       ``meta/info.json`` AFTER ``stop_recording`` returns, NOT
       self-reported by the loop. Mismatch with ``n_episodes`` is surfaced
       as ``warnings=[...]`` for the verifier to act on.

    Args:
        simulation: Live ``Simulation`` (or compatible) handle.
            Constructed by the orchestrator - pass through a Python
            partial / closure, not from agent text. LLMs cannot
            synthesize this argument, which is the point: the episode
            loop runs in deterministic Python.
        robot_name: Robot to control. Forwarded to ``run_policy``.
            Required when the simulation hosts more than one robot.
        policy_provider: Provider name passed to ``create_policy``
            inside the engine (``"mock"`` / ``"lerobot_local"`` /
            ``"groot"`` / ``"molmoact2"`` / ...).
        policy_config: Provider-specific kwargs forwarded verbatim.
        instruction: Natural-language instruction for the policy.
        n_episodes: Number of reset -> rollout episodes. MUST be a
            positive int. There is no "guess from duration" fallback.
        n_steps: Hard cap on control steps per episode. Forwarded to
            ``run_policy`` as ``n_steps``.
        control_frequency: Target Hz for policy queries.
        action_horizon: Actions consumed per policy call before
            re-querying.
        fast_mode: Skip real-time sleep between steps (default True for
            rollouts - wall-clock pacing slows headless eval).
        dataset_root: When set, the tool drives the full recording
            cycle: ``start_recording(root=dataset_root, ...)`` -> N
            rollouts with per-episode save_episode -> ``stop_recording``
            -> parquet-truth read. When ``None`` the loop runs without
            recording (smoke-test mode).
        dataset_repo_id: Forwarded to ``start_recording``.
        dataset_task: Task label forwarded to ``start_recording``.
        dataset_fps: Dataset FPS forwarded to ``start_recording``.
        seed: Master RNG seed. Each episode derives a deterministic
            offset so rollouts are reproducible within a process.
        policy_kwargs: Optional per-call goal payload forwarded to
            every ``policy.get_actions`` call (the #300 goal keys).

    Returns:
        Standard ``{status, content}`` payload. On success the payload
        also carries::

            {
                "n_episodes_requested": int,
                "n_episodes_actual": int,      # parquet-truth
                "n_frames_actual": int,        # parquet-truth
                "dataset_root": str | None,
                "warnings": [str, ...],        # mismatch flags
                "episodes": [
                    {"index": int, "status": "success" | "error", ...},
                    ...
                ],
            }
    """
    # ---- 1. Validation ---------------------------------------------------
    if simulation is None:
        return _err(
            "run_policy: `simulation` is required (pass the live Simulation/Robot "
            "handle from the orchestrator). LLMs cannot synthesize this argument "
            "- that is the point: the episode loop must run in deterministic Python. "
            "See #708 for the fabrication vector this tool closes."
        )

    if not hasattr(simulation, "run_policy"):
        return _err(
            f"run_policy: `simulation` of type {type(simulation).__name__!r} does "
            "not expose .run_policy(). Pass a strands_robots Simulation or "
            "compatible engine."
        )

    if not isinstance(n_episodes, int) or isinstance(n_episodes, bool) or n_episodes < 1:
        return _err(
            f"run_policy: n_episodes must be a positive int, got {n_episodes!r}. "
            "This loop iterates exactly n_episodes times; there is no fallback."
        )

    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        return _err(f"run_policy: n_steps must be a positive int, got {n_steps!r}.")

    # ---- 2. Optional: start recording -----------------------------------
    recording_started = False
    if dataset_root is not None:
        if not hasattr(simulation, "start_recording"):
            return _err(
                "run_policy: dataset_root requested but simulation does not "
                "expose .start_recording(). Install the [lerobot] extra or pass "
                "dataset_root=None for a recording-less rollout."
            )
        start_result = simulation.start_recording(
            repo_id=dataset_repo_id,
            task=dataset_task,
            fps=dataset_fps,
            root=dataset_root,
            overwrite=True,
        )
        if start_result.get("status") != "success":
            # Surface the engine's own error verbatim - it already explains
            # missing extras, world not loaded, etc.
            return start_result
        recording_started = True

    # ---- 3. Episode loop ------------------------------------------------
    episodes: list[dict[str, Any]] = []
    try:
        for ep in range(n_episodes):
            ep_seed = None if seed is None else seed + ep
            try:
                rollout = simulation.run_policy(
                    robot_name=robot_name,
                    policy_provider=policy_provider,
                    policy_config=policy_config,
                    instruction=instruction,
                    control_frequency=control_frequency,
                    action_horizon=action_horizon,
                    fast_mode=fast_mode,
                    n_steps=n_steps,
                    max_steps=n_steps,
                    policy_kwargs=policy_kwargs,
                    seed=ep_seed,
                )
            except Exception as e:  # noqa: BLE001 - per-episode resilience
                logger.exception("Episode %d/%d raised: %s", ep + 1, n_episodes, e)
                rollout = {
                    "status": "error",
                    "content": [{"text": f"Episode {ep + 1} raised: {e!r}"}],
                }

            episodes.append(
                {
                    "index": ep,
                    "status": rollout.get("status", "error"),
                    "text": (rollout.get("content") or [{}])[0].get("text", "")[:500],
                }
            )

            # Per-episode parquet boundary. PR #716 wired this helper inside
            # PolicyRunner.evaluate() / _evaluate_with_spec(), but bare
            # run_policy does NOT call it (single-rollout APIs assume the
            # caller owns episode framing). We do it here because we ARE the
            # caller.
            #
            # The helper lives on PolicyRunner. We get the active runner
            # via the simulation's policy-thread bookkeeping OR construct a
            # lightweight wrapper that reads the recorder out of
            # ``sim._world._backend_state``. The simplest path that respects
            # the existing API surface: build a transient PolicyRunner just
            # for the finalize call. It's stateless w.r.t. the recorder.
            if recording_started:
                _finalize_episode(simulation)

    finally:
        # ---- 4. Stop recording (always, on success or failure) ----------
        # idempotent on the simulation side ("Was not recording." path), so
        # safe to call even if the start_recording above failed silently.
        if recording_started and hasattr(simulation, "stop_recording"):
            stop_result = simulation.stop_recording()
            if stop_result.get("status") != "success":
                logger.warning(
                    "run_policy: stop_recording returned non-success: %s",
                    stop_result,
                )

    # ---- 5. Parquet-truth gate -----------------------------------------
    n_actual_eps = -1
    n_actual_frames = -1
    warnings_: list[str] = []
    truth: dict[str, Any] = {}

    if dataset_root is not None:
        truth = _read_parquet_truth(dataset_root)
        if not truth.get("info_present"):
            warnings_.append(
                f"meta/info.json missing under {dataset_root!r} - cannot "
                "verify episode count from parquet truth. "
                f"({truth.get('error', 'no error reported')})"
            )
        else:
            n_actual_eps = truth["total_episodes"]
            n_actual_frames = truth["total_frames"]
            if n_actual_eps != n_episodes:
                warnings_.append(
                    f"FABRICATION GUARD: requested {n_episodes} episodes, "
                    f"meta/info.json:total_episodes={n_actual_eps}. The "
                    "per-episode save_episode boundary did not fire as "
                    "expected. See #708."
                )

    # ---- 6. Build payload ----------------------------------------------
    n_ok = sum(1 for e in episodes if e["status"] == "success")
    summary_line = f"run_policy: {n_ok}/{n_episodes} episodes ok" + (
        f" | parquet-truth: total_episodes={n_actual_eps}, total_frames={n_actual_frames}"
        if dataset_root is not None
        else ""
    )
    if warnings_:
        summary_line += f" | warnings={len(warnings_)}"

    payload = {
        "n_episodes_requested": n_episodes,
        "n_episodes_actual": n_actual_eps,
        "n_frames_actual": n_actual_frames,
        "n_episodes_ok": n_ok,
        "dataset_root": dataset_root,
        "warnings": warnings_,
        "episodes": episodes,
    }
    if truth.get("info_path"):
        payload["info_path"] = truth["info_path"]

    out_status = "success" if (n_ok == n_episodes and not warnings_) else "error"
    return {
        "status": out_status,
        "content": [{"text": summary_line}, {"json": payload}],
    }


def _finalize_episode(simulation: Any) -> None:
    """Invoke ``PolicyRunner._finalize_recorder_episode`` for ``simulation``.

    PR #716 added this helper as the canonical per-episode boundary on
    ``PolicyRunner`` (it reads the active recorder out of
    ``sim._world._backend_state["dataset_recorder"]`` and calls its
    ``save_episode``). Bare ``run_policy`` does not invoke it - it assumes
    the caller owns episode framing. Since this tool *is* the caller, we
    delegate to the same helper to keep the boundary logic in one place
    and to inherit its tolerance for absent/empty buffers and save errors.
    """
    try:
        from strands_robots.simulation.policy_runner import PolicyRunner
    except ImportError:
        logger.debug("PolicyRunner unavailable; skipping per-episode finalize")
        return

    try:
        runner = PolicyRunner(simulation)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not construct PolicyRunner for finalize: %s", e)
        return

    try:
        runner._finalize_recorder_episode()  # noqa: SLF001 - this is the contract surface
    except Exception as e:  # noqa: BLE001
        logger.warning("Per-episode finalize raised: %s", e)
