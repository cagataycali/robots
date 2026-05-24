#!/usr/bin/env python3
"""Rogue 12 -- RPC response hijack (D1).

Threat model:
  Operator A sends a point-to-point RPC to robot R via
  ``Mesh.send(target="R", ...)``. The wire-level ACL gates the
  ``cmd`` channel, but several peers can also legitimately publish on
  ``response/**`` (replies are broadcast on the channel). An
  ACL-authorised peer could observe operator A's ``turn_id`` and
  publish a forged response on it -- the sender would accept the
  forged result.

Defence under test:
  ``Mesh._on_response`` cross-checks ``responder_id`` against the
  expected target stored in ``_expected_responders[turn_id]``. A
  mismatch is dropped and audited as ``response_hijack_rejected``.

This is a textbook *insider* attack (everyone has wire-layer
identity already). The defence is the only thing standing between
lateral mischief and result-injection.
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
    rogue_id = "rogue_12_response_hijacker"
    av_id = "AV-32"
    title = "Forged response on a P2P turn dropped at responder_id check"
    posture = "in-process; pending point-to-point send to robot R"

    t0 = time.time()
    held = False
    observed = ""
    error = ""
    outcomes: list[tuple[str, bool]] = []

    try:
        import threading
        from strands_robots.mesh.core import BROADCAST_RESPONDER, Mesh

        sender = Mesh(robot=_StubRobot(), peer_id="operator-1")
        # Manually register a pending P2P turn -- this mirrors what
        # Mesh.send does internally before publishing the cmd.
        turn = "turn-0001-aabbcc"
        with sender._rpc_lock:
            sender._pending[turn] = threading.Event()
            sender._expected_responders[turn] = "R"  # legit target
            sender._responses[turn] = []

        # Attack: a different ACL-authorised peer publishes a response
        # forging the turn_id but with their own responder_id.
        forged = {"turn_id": turn, "responder_id": "hijacker",
                  "result": {"status": "locked"}}
        sender._on_response(_sample(forged))
        forged_dropped = len(sender._responses.get(turn, [])) == 0
        outcomes.append(("forged_response_dropped", forged_dropped))

        # Legit response from R is accepted.
        legit = {"turn_id": turn, "responder_id": "R",
                 "result": {"status": "idle"}}
        sender._on_response(_sample(legit))
        legit_accepted = len(sender._responses.get(turn, [])) == 1
        outcomes.append(("legit_response_accepted", legit_accepted))

        # Broadcast turn: any responder accepted.
        broadcast_turn = "turn-bcast-001"
        with sender._rpc_lock:
            sender._pending[broadcast_turn] = threading.Event()
            sender._expected_responders[broadcast_turn] = BROADCAST_RESPONDER
            sender._responses[broadcast_turn] = []
        any_responder = {"turn_id": broadcast_turn, "responder_id": "any-peer",
                          "result": {"status": "idle"}}
        sender._on_response(_sample(any_responder))
        outcomes.append(("broadcast_accepts_any",
                         len(sender._responses.get(broadcast_turn, [])) == 1))

        # No matching turn -> dropped
        unknown = {"turn_id": "turn-no-such", "responder_id": "x", "result": {}}
        sender._on_response(_sample(unknown))
        outcomes.append(("unknown_turn_ignored",
                         "turn-no-such" not in sender._responses))

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
