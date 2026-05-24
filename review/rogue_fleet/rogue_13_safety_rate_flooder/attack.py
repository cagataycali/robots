#!/usr/bin/env python3
"""Rogue 13 -- novel-t safety estop flood (per-issuer fairness).

Threat model:
  Even with the replay cache, an attacker who can mint *novel* `t`
  values (e.g. by being authorised on the safety topic) could flood
  the receiver with a stream of legitimately-fresh estop envelopes,
  each of which engages then re-engages the lockout, exhausting the
  per-issuer audit / cache budget and crowding out other operators'
  legitimate envelopes.

Defences:
  1. Transport-level ``downsampling`` rule on ``safety/**`` at
     ``STRANDS_MESH_SAFETY_RATE_HZ`` (default 2 Hz).
  2. F8-A / F9-A: per-issuer fairness cap derived from the replay
     cache contents. After a single issuer fills the
     ``STRANDS_MESH_RESUME_REPLAY_CACHE_MAX``-size budget, additional
     novel-t envelopes from the same issuer are dropped at the
     receiver.

This rogue exercises (2) directly (in-process); (1) is an external
Zenoh interceptor that we touch in rogue_04.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rogue_fleet._lib.report import RogueResult, write_result  # noqa: E402


class _StubRobot:
    def get_task_status(self): return {"status": "idle"}


def _sample(env: dict):
    return SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda: json.dumps(env).encode())
    )


def main() -> int:
    rogue_id = "rogue_13_safety_rate_flooder"
    av_id = "AV-SAFETY-RATE"
    title = "Per-issuer fairness limits the novel-t flood from one peer"
    posture = "in-process; STRANDS_MESH_RESUME_REPLAY_CACHE_MAX=8"

    t0 = time.time()
    held = False
    observed = ""
    error = ""

    try:
        # Constrain the cache so we can prove fairness in finite time.
        os.environ["STRANDS_MESH_RESUME_REPLAY_CACHE_MAX"] = "8"
        from strands_robots.mesh.core import (
            Mesh,
            _resume_replay_cache_max,
        )
        cache_max = _resume_replay_cache_max()

        m = Mesh(robot=_StubRobot(), peer_id="victim-r1")

        # One issuer floods novel-t envelopes well past the cache max.
        flood_count = cache_max * 4
        engaged_for_flooder = 0
        for i in range(flood_count):
            m._estop_lockout.clear()
            t_i = time.time() + (i * 0.001)
            m._on_safety_estop(_sample({
                "peer_id": "FLOODER", "t": t_i, "type": "estop",
            }))
            if m._estop_lockout.is_set():
                engaged_for_flooder += 1

        # A second, well-behaved operator sends ONE legitimate envelope.
        # If the cache is per-issuer fair, this one still engages.
        m._estop_lockout.clear()
        m._on_safety_estop(_sample({
            "peer_id": "good-operator", "t": time.time() + 1.0,
            "type": "estop",
        }))
        good_engaged = m._estop_lockout.is_set()

        # Per-issuer cap should bound flooder's effective engagements.
        # We don't pin an exact number (depends on review-cycle tuning)
        # but we DO assert (a) flooder did not engage on every iteration
        # (some were dropped by fairness) and (b) the good operator
        # still got their lockout fire.
        flooder_was_throttled = engaged_for_flooder < flood_count
        held = flooder_was_throttled and good_engaged
        observed = (
            f"cache_max={cache_max} flood_attempts={flood_count} "
            f"flooder_engaged={engaged_for_flooder} "
            f"good_op_engaged={good_engaged} "
            f"flooder_throttled={flooder_was_throttled}"
        )
    except Exception:  # noqa: BLE001
        error = traceback.format_exc()
        observed = "unexpected exception in rogue"

    write_result(
        RogueResult(
            rogue_id=rogue_id, av_id=av_id, title=title, posture=posture,
            defence_held=held, observed=observed, error=error,
            duration_s=time.time() - t0,
        )
    )
    return 0 if held else 1


if __name__ == "__main__":
    raise SystemExit(main())
