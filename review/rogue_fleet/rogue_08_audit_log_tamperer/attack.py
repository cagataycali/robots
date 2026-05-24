#!/usr/bin/env python3
"""Rogue 08 -- audit-log tamper detection.

Threat model:
  Post-incident, an attacker with file-system access tries to scrub their
  tracks in the audit log:

    1. Edit one record's payload -- breaks HMAC sig (bad_signature).
    2. Delete a record -- causes a seq gap (sequence_gaps).
    3. Append an unsigned record to a signed log (missing_sig).
    4. Rotate the PSK -- new records sign under PSK-B; verifying with
       PSK-A flags them as bad_signature.

Defence under test:
  ``audit.verify_audit_integrity`` returns ``ok=False`` and points
  at every variant.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rogue_fleet._lib.report import RogueResult, write_result  # noqa: E402


def main() -> int:
    rogue_id = "rogue_08_audit_log_tamperer"
    av_id = "AV-21+22+23+24+25"
    title = "Audit log tamper detection holds across 4 attack variants"
    posture = "in-process; isolated audit dir; PSK-signed log"

    t0 = time.time()
    held = False
    observed = ""
    error = ""
    outcomes: list[tuple[str, bool]] = []

    try:
        from strands_robots.mesh import audit

        with tempfile.TemporaryDirectory(prefix="audit_tamper_") as td:
            audit_dir = Path(td)
            os.environ["STRANDS_MESH_AUDIT_DIR"] = str(audit_dir)
            os.environ["STRANDS_MESH_AUDIT_PSK"] = "psk-A"
            importlib.reload(audit)

            log_path = audit_dir / "mesh_audit.jsonl"

            # === Attack 1: HMAC tamper ===
            for i in range(4):
                audit.log_safety_event("command_received", "victim-r1",
                                       {"i": i, "action": "status"})
            lines = log_path.read_text().splitlines()
            tampered = json.loads(lines[1])
            tampered["payload"]["i"] = 999
            lines[1] = json.dumps(tampered)
            log_path.write_text("\n".join(lines) + "\n")
            report = audit.verify_audit_integrity()
            outcomes.append(("hmac_tamper_detected",
                             report.get("bad_signature", 0) > 0
                             and not report.get("ok", True)))

            # === Attack 2: seq gap (delete a record) ===
            log_path.unlink()
            importlib.reload(audit)
            for i in range(4):
                audit.log_safety_event("command_received", "victim-r1",
                                       {"i": i})
            lines = log_path.read_text().splitlines()
            del lines[1]
            log_path.write_text("\n".join(lines) + "\n")
            report = audit.verify_audit_integrity()
            outcomes.append(("seq_gap_detected",
                             len(report.get("sequence_gaps", [])) > 0
                             and not report.get("ok", True)))

            # === Attack 3: append unsigned record ===
            log_path.unlink()
            importlib.reload(audit)
            for i in range(2):
                audit.log_safety_event("x", "victim-r1", {"i": i})
            unsigned = {
                "ts": time.time(), "event": "unsigned_replacement",
                "peer_id": "victim-r1", "payload": {"i": 99}, "seq": 99,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(unsigned) + "\n")
            report = audit.verify_audit_integrity()
            outcomes.append(("unsigned_degrade_detected",
                             report.get("missing_sig", 0) > 0
                             and not report.get("ok", True)))

            # === Attack 4: PSK rotation A->B; verify with A ===
            log_path.unlink()
            importlib.reload(audit)
            for i in range(2):
                audit.log_safety_event("x", "victim-r1", {"i": i})
            os.environ["STRANDS_MESH_AUDIT_PSK"] = "psk-B"
            importlib.reload(audit)
            try:
                audit.log_safety_event("x", "victim-r1", {"i": 99})
            except Exception as e:  # noqa: BLE001
                # PSK rotation may itself raise (R4-2 degraded path);
                # that's also a valid "detected" signal.
                observed += f" (rotation-raise: {type(e).__name__})"
            os.environ["STRANDS_MESH_AUDIT_PSK"] = "psk-A"
            importlib.reload(audit)
            report = audit.verify_audit_integrity()
            outcomes.append(("psk_rotation_detected",
                             report.get("bad_signature", 0) > 0
                             or report.get("missing_sig", 0) > 0
                             or not report.get("ok", True)))

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
