#!/usr/bin/env python3
"""Rogue 05 -- safety estop replay & peer_id permutation.

Threat model:
  Attacker has captured a legitimate ``estop`` envelope from a previous
  fleet incident (read off the bus, lifted from the audit log of a
  decommissioned peer, etc.). They want to weaponise it -- replay the
  envelope to lock down a victim robot at will, or permute the
  ``peer_id`` field to create the appearance of multiple distinct
  emergency-stop sources to evade per-issuer fairness limits.

Defences under test:
  1. ``Mesh._estop_replay_cache`` keyed by ``t`` (timestamp): a
     second arrival of an envelope with the same ``t`` is dropped
     (AV-26).
  2. The cache deliberately keys on ``t`` only -- permuting the
     payload ``peer_id`` does NOT escape the cache (AV-27).
  3. F8-A / F9-A: per-issuer fairness derived from cache contents,
     not a separate ``_estop_replay_per_issuer`` dict. We exercise
     the modern shape.

This rogue is needs_victim=False because the replay-cache logic is
entirely receiver-side -- we instantiate a Mesh instance in this
process against a stub robot, drive ``_on_safety_estop`` directly
with forged samples, and inspect the lockout state. This mirrors
AV-26 / AV-27 in the unit-style harness but lives in its own pid so
it can be promoted to a multi-host fleet test in future.
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
    def get_task_status(self):
        return {"status": "idle"}


def _fake_sample(envelope: dict):
    """Forge a Zenoh sample-shaped object whose ``payload.to_bytes()`` is JSON."""
    payload = SimpleNamespace(to_bytes=lambda: json.dumps(envelope).encode())
    return SimpleNamespace(payload=payload)


def main() -> int:
    rogue_id = "rogue_05_safety_replay_attacker"
    av_id = "AV-26+27"
    title = "Safety estop replay + peer_id permutation blocked by t-keyed cache"
    posture = "in-process Mesh; receiver-side _estop_replay_cache active"

    t0 = time.time()
    held = False
    observed = ""
    error = ""

    try:
        from strands_robots.mesh.core import Mesh

        mesh = Mesh(robot=_StubRobot(), peer_id="victim-r1")
        now = time.time()

        # Attack 1: replay the SAME envelope (same t).
        env1 = {"peer_id": "operator-1", "t": now, "type": "estop"}
        mesh._on_safety_estop(_fake_sample(env1))
        first_engaged = mesh._estop_lockout.is_set()
        # Operator clears lockout (legit resume); cache must persist.
        mesh._estop_lockout.clear()
        mesh._on_safety_estop(_fake_sample(env1))
        second_engaged = mesh._estop_lockout.is_set()
        replay_blocked = first_engaged and not second_engaged

        # Attack 2: permute the peer_id field, keep t identical. The
        # cache keys on t alone, so this still gets blocked.
        env2 = {"peer_id": "OPERATOR-2", "t": now, "type": "estop"}
        mesh._on_safety_estop(_fake_sample(env2))
        permutation_engaged = mesh._estop_lockout.is_set()
        permutation_blocked = not permutation_engaged

        # Attack 3: novel-t replay (the cache only blocks repeats of
        # known t values). A genuine new t fires lockout. We assert
        # the legitimate path still works.
        mesh._estop_lockout.clear()
        env3 = {"peer_id": "operator-3", "t": now + 0.1, "type": "estop"}
        mesh._on_safety_estop(_fake_sample(env3))
        legit_engaged = mesh._estop_lockout.is_set()

        held = replay_blocked and permutation_blocked and legit_engaged
        observed = (
            f"replay_blocked={replay_blocked} "
            f"permutation_blocked={permutation_blocked} "
            f"legit_engaged={legit_engaged}"
        )
    except Exception:  # noqa: BLE001
        error = traceback.format_exc()
        observed = "unexpected exception in rogue (see error)"

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
