#!/usr/bin/env python3
"""Rogue 11 -- safety/resume token forgery attempts.

Threat model:
  An attacker who reaches the safety/resume topic (got past mTLS+ACL)
  tries to clear an estop lockout without knowing the operator
  override code. Three variants:

  1. No override_code/proof at all -- envelope shape gates.
  2. Wrong HMAC -- compute proof under attacker's guessed code.
  3. Missing proof_nonce -- skip the freshness binding.

Also verifies the **fail-closed default**: if the receiver lacks
``STRANDS_MESH_OVERRIDE_CODE``, it refuses every remote resume.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
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
    def get_task_status(self): return {"status": "locked"}


def _sample(env: dict):
    return SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda: json.dumps(env).encode())
    )


def _engage(mesh):
    """Force the mesh into lockout with a legit estop envelope."""
    mesh._on_safety_estop(_sample({
        "peer_id": "operator-1", "t": time.time(), "type": "estop",
    }))


def main() -> int:
    rogue_id = "rogue_11_resume_token_replay"
    av_id = "AV-RESUME-HMAC"
    title = "Resume token: missing/bad HMAC + missing nonce all rejected"
    posture = "in-process; STRANDS_MESH_OVERRIDE_CODE='real-secret'"

    t0 = time.time()
    held = False
    observed = ""
    error = ""
    outcomes: list[tuple[str, bool]] = []

    try:
        # Fail-closed test FIRST: without OVERRIDE_CODE, every resume rejected.
        os.environ.pop("STRANDS_MESH_OVERRIDE_CODE", None)
        from strands_robots.mesh.core import Mesh
        m = Mesh(robot=_StubRobot(), peer_id="victim-r1")
        _engage(m)
        m._on_safety_resume(_sample({
            "peer_id": "operator-1", "t": time.time(),
            "override_proof": "any", "proof_nonce": "x",
        }))
        outcomes.append(("fail_closed_no_local_code", m._estop_lockout.is_set()))

        # Now configure the local override code
        os.environ["STRANDS_MESH_OVERRIDE_CODE"] = "real-secret"

        # 1. Missing override_proof + proof_nonce
        m = Mesh(robot=_StubRobot(), peer_id="victim-r1")
        _engage(m)
        m._on_safety_resume(_sample({
            "peer_id": "operator-1", "t": time.time(), "type": "resume",
        }))
        outcomes.append(("missing_proof_blocked", m._estop_lockout.is_set()))

        # 2. Wrong HMAC (attacker guesses code)
        m = Mesh(robot=_StubRobot(), peer_id="victim-r1")
        _engage(m)
        nonce = "ABCDEF"
        wrong = _hmac.new(b"wrong-guess", nonce.encode(), hashlib.sha256).hexdigest()
        m._on_safety_resume(_sample({
            "peer_id": "operator-1", "t": time.time(),
            "proof_nonce": nonce, "override_proof": wrong, "type": "resume",
        }))
        outcomes.append(("wrong_hmac_blocked", m._estop_lockout.is_set()))

        # 3. Missing proof_nonce
        m = Mesh(robot=_StubRobot(), peer_id="victim-r1")
        _engage(m)
        m._on_safety_resume(_sample({
            "peer_id": "operator-1", "t": time.time(),
            "override_proof": "deadbeef", "type": "resume",
        }))
        outcomes.append(("missing_nonce_blocked", m._estop_lockout.is_set()))

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
