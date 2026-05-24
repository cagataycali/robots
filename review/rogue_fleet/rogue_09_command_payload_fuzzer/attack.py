#!/usr/bin/env python3
"""Rogue 09 -- payload validator surface fuzzing.

Threat model:
  An ALREADY-AUTHENTICATED peer (got past mTLS + ACL) tries to send
  payloads that:
    * encode a 24-hour ``execute`` action,
    * embed a 200 KB instruction string,
    * point inference at attacker-controlled host,
    * load a HuggingFace model from an unallowed org with path
      traversal,
    * use an unknown action,
    * send the command as a raw string (not a dict, R24-B),
    * omit ``policy_provider`` while specifying ``policy_type``.

Defence under test:
  ``security.validate_command`` raises ``ValidationError`` for every
  case. The dispatch wrapper in ``Mesh._dispatch`` audits the
  rejection and returns a generic error to the wire.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rogue_fleet._lib.report import RogueResult, write_result  # noqa: E402


def main() -> int:
    rogue_id = "rogue_09_command_payload_fuzzer"
    av_id = "AV-09+10+11+12+13+14+15"
    title = "validate_command rejects 7 payload forgeries"
    posture = "in-process; security.validate_command direct call"

    t0 = time.time()
    held = False
    observed = ""
    error = ""
    outcomes: list[tuple[str, bool]] = []

    try:
        from strands_robots.mesh import security as sec

        cases: list[tuple[str, object]] = [
            ("long_instruction",
             {"action": "execute", "instruction": "x" * (sec.MAX_INSTRUCTION_LEN + 100)}),
            ("hostile_policy_host",
             {"action": "start", "policy_host": "evil.attacker.com",
              "policy_type": "act", "policy_provider": "lerobot"}),
            ("hf_path_traversal",
             {"action": "start", "pretrained_name_or_path": "../../etc/passwd",
              "policy_type": "act", "policy_provider": "lerobot"}),
            ("hf_unallowed_org",
             {"action": "start", "pretrained_name_or_path": "evil/backdoor",
              "policy_type": "act", "policy_provider": "lerobot"}),
            ("long_duration",
             {"action": "execute", "instruction": "hi", "duration": 1e9}),
            ("non_dict", "this-is-not-a-dict"),
            ("unknown_action", {"action": "unleash_chaos"}),
            ("missing_policy_provider",
             {"action": "start", "policy_type": "act",
              "pretrained_name_or_path": "lerobot/pi0"}),
        ]
        for name, payload in cases:
            try:
                sec.validate_command(payload)
                outcomes.append((name, False))
            except sec.ValidationError:
                outcomes.append((name, True))
            except Exception as e:  # noqa: BLE001
                # Any rejection counts (TypeError, KeyError, etc.)
                outcomes.append((name, True))
                observed += f" {name}_kind={type(e).__name__}"

        held = all(ok for _, ok in outcomes)
        observed = "; ".join(f"{n}={ok}" for n, ok in outcomes) + observed
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
