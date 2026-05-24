#!/usr/bin/env python3
"""Rogue 06 -- safety envelope freshness & shape attacks.

Threat model:
  Attacker either (a) forges an envelope from scratch (knows the JSON
  shape but not the freshness rules), or (b) replays an envelope that
  is fresh-window-stale, or (c) tries time-skew shenanigans by setting
  `t` to the far future to keep an estop "valid" indefinitely.

Defences under test:
  * `core.Mesh._on_safety_estop` rejects envelopes:
    - missing `t` (AV-28)
    - `t` older than ``STRANDS_MESH_RESUME_FRESHNESS_S`` (AV-29)
    - `t` skewed forward beyond ``STRANDS_MESH_RESUME_FORWARD_SKEW_S`` (AV-30)
    - missing / empty `peer_id` (AV-31)

All four are receiver-side payload checks executed before the
lockout fires.
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
    rogue_id = "rogue_06_safety_envelope_forger"
    av_id = "AV-28+29+30+31"
    title = "Estop envelope freshness/shape gates reject 4 forgery variants"
    posture = "in-process Mesh; testing missing/stale/forward t + missing peer_id"

    t0 = time.time()
    held = False
    observed = ""
    error = ""

    try:
        from strands_robots.mesh.core import Mesh, _resume_freshness_window_s

        freshness = _resume_freshness_window_s()
        outcomes: list[tuple[str, bool]] = []

        # 1. Missing `t`
        m = Mesh(robot=_StubRobot(), peer_id="victim-r1")
        m._on_safety_estop(_sample({"peer_id": "op", "type": "estop"}))
        outcomes.append(("missing_t_blocked", not m._estop_lockout.is_set()))

        # 2. Stale `t`
        m = Mesh(robot=_StubRobot(), peer_id="victim-r1")
        stale = time.time() - (freshness * 5)
        m._on_safety_estop(_sample({"peer_id": "op", "t": stale, "type": "estop"}))
        outcomes.append(("stale_t_blocked", not m._estop_lockout.is_set()))

        # 3. Forward-skewed `t` (way in the future)
        m = Mesh(robot=_StubRobot(), peer_id="victim-r1")
        future = time.time() + (freshness * 100)
        m._on_safety_estop(_sample({"peer_id": "op", "t": future, "type": "estop"}))
        outcomes.append(("forward_skew_blocked", not m._estop_lockout.is_set()))

        # 4. Missing peer_id
        m = Mesh(robot=_StubRobot(), peer_id="victim-r1")
        m._on_safety_estop(_sample({"t": time.time(), "type": "estop"}))
        outcomes.append(("missing_peer_id_blocked", not m._estop_lockout.is_set()))

        # 5. Empty peer_id (string "")
        m = Mesh(robot=_StubRobot(), peer_id="victim-r1")
        m._on_safety_estop(_sample({"peer_id": "", "t": time.time(), "type": "estop"}))
        outcomes.append(("empty_peer_id_blocked", not m._estop_lockout.is_set()))

        held = all(ok for _, ok in outcomes)
        observed = "; ".join(f"{n}={ok}" for n, ok in outcomes)
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
