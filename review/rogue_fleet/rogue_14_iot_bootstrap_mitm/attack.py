#!/usr/bin/env python3
"""Rogue 14 -- IoT bootstrap CA-pin MITM.

Threat model:
  When a robot bootstraps via AWS IoT it downloads ``AmazonRootCA1.pem``
  over HTTPS. A network-level adversary (DNS hijack, captive portal,
  BGP route attack, malicious corporate proxy) can substitute a rogue
  CA at the canonical URL. Without a pinned hash, the bootstrap
  silently trusts the rogue chain.

  Two adjacent attack patterns:

  1. **Existing-file rogue** -- attacker plants a rogue CA at the
     canonical on-disk path BEFORE bootstrap runs. Hopes the code
     re-uses the file without re-checking the pin.
  2. **Env-var bypass** -- attacker hopes
     ``STRANDS_MESH_DISABLE_CA_PIN=true`` lets the existing-file path
     accept any bytes. The audit explicitly documents this should not
     work for the existing-file branch.

Defences under test:
  ``provision._verify_ca_bytes`` and the existing-file path in the
  bootstrap helper. The pin set is built-in (`_AMAZON_ROOT_CA1_PINS`)
  and extensible via ``STRANDS_MESH_CA_PINS``.
"""

from __future__ import annotations

import hashlib
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
    rogue_id = "rogue_14_iot_bootstrap_mitm"
    av_id = "AV-IOT-CA-PIN"
    title = "IoT bootstrap rejects rogue CA bytes; env-var bypass scoped"
    posture = "in-process; provision._verify_ca_bytes & existing-file branch"

    t0 = time.time()
    held = False
    observed = ""
    error = ""
    outcomes: list[tuple[str, bool]] = []

    try:
        from strands_robots.mesh.iot import provision as p

        # Built-in pin set is non-empty and contains 64-char lowercase hex
        pins = p._resolve_ca_pins()
        outcomes.append(("pin_set_nonempty", len(pins) > 0))
        outcomes.append(("pin_format_valid",
                          all(len(x) == 64 and x.islower() and
                              all(c in '0123456789abcdef' for c in x)
                              for x in pins)))

        # Rogue bytes don't match the pin -> _verify_ca_bytes returns False
        rogue_bytes = b"-----BEGIN CERTIFICATE-----\nROGUEROGUEROGUE\n-----END CERTIFICATE-----\n"
        outcomes.append(("rogue_bytes_rejected",
                          not p._verify_ca_bytes(rogue_bytes)))

        # _hash_matches_pin works the same way (used by existing-file branch)
        outcomes.append(("rogue_hash_no_match",
                          not p._hash_matches_pin(rogue_bytes)))

        # The existing-file branch refuses to re-use a rogue CA even
        # with STRANDS_MESH_DISABLE_CA_PIN=true. Simulate by writing
        # rogue bytes to a tempfile and calling the bootstrap helper.
        with tempfile.TemporaryDirectory(prefix="iot_ca_") as td:
            ca_path = Path(td) / "AmazonRootCA1.pem"
            ca_path.write_bytes(rogue_bytes)
            os.environ["STRANDS_MESH_DISABLE_CA_PIN"] = "true"
            try:
                p._ensure_ca(ca_path)
                outcomes.append(("env_bypass_does_not_apply_to_existing_file", False))
            except (RuntimeError, ValueError):
                outcomes.append(("env_bypass_does_not_apply_to_existing_file", True))
            os.environ.pop("STRANDS_MESH_DISABLE_CA_PIN", None)

            # Operator extension via STRANDS_MESH_CA_PINS
            extra = hashlib.sha256(b"trusted-extension").hexdigest()
            os.environ["STRANDS_MESH_CA_PINS"] = extra
            extended = p._resolve_ca_pins()
            outcomes.append(("env_extension_picked_up", extra in extended))
            os.environ.pop("STRANDS_MESH_CA_PINS", None)

            # Malformed pin (not 64-char hex) -> dropped with WARNING, not accepted
            os.environ["STRANDS_MESH_CA_PINS"] = "not-a-hash,;rm -rf /"
            extended = p._resolve_ca_pins()
            outcomes.append(("malformed_pin_dropped",
                              all(len(x) == 64 for x in extended)))
            os.environ.pop("STRANDS_MESH_CA_PINS", None)

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
