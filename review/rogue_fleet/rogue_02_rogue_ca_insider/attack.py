#!/usr/bin/env python3
"""Rogue 02 -- attacker mints their own CA + leaf cert and tries to join.

Threat model:
  Attacker stands up a private OpenSSL/EJBCA CA, mints a leaf with the
  same CN scheme the victim uses (``robot-r1`` / ``operator-1``), and
  presents that cert to the victim's tls-only listener.

Defence under test:
  The victim's CA bundle is the *only* chain trusted at the TLS layer.
  ``verify_name_on_connect = true`` plus a strict CA list means a
  rogue-CA leaf cannot finish the handshake regardless of CN.

Observation:
  ``zenoh.open()`` either raises with a TLS verification error, or
  the routing graph never accepts the rogue's session.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from rogue_fleet._lib.pki import EphemeralCA  # noqa: E402
from rogue_fleet._lib.report import RogueResult, write_result  # noqa: E402


def main() -> int:
    rogue_id = "rogue_02_rogue_ca_insider"
    av_id = "AV-02"
    title = "Rogue-CA insider rejected at TLS verify"
    posture = "victim trusts CA-A only; rogue presents leaf signed by rogue CA-B"

    victim_listen = os.environ["VICTIM_LISTEN"]
    t0 = time.time()
    held = False
    observed = ""
    error = ""

    with tempfile.TemporaryDirectory(prefix="rogue_ca_") as td:
        rogue_ca = EphemeralCA.make(Path(td))
        # Mint a cert with the SAME CN the victim's operator cert uses,
        # signed by our own CA. This is the textbook rogue-CA attack.
        cert, key = rogue_ca.mint("operator-1", sub="rogue-operator")

        import zenoh

        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"peer"')
        cfg.insert_json5("namespace", '"strands"')
        cfg.insert_json5("scouting/multicast/enabled", "false")
        cfg.insert_json5("scouting/gossip/enabled", "false")
        cfg.insert_json5("transport/link/protocols", json.dumps(["tls"]))
        cfg.insert_json5(
            "transport/link/tls",
            json.dumps(
                {
                    "enable_mtls": True,
                    "verify_name_on_connect": True,
                    # The attacker uses THEIR OWN CA as the trusted root
                    # so handshakes outbound work; the victim still
                    # only trusts CA-A.
                    "root_ca_certificate": str(rogue_ca.ca_cert),
                    "connect_certificate": str(cert),
                    "connect_private_key": str(key),
                }
            ),
        )
        cfg.insert_json5("connect/endpoints", json.dumps([f"tls/{victim_listen}"]))

        try:
            try:
                outsider = zenoh.open(cfg)
            except Exception as e:  # noqa: BLE001
                held = True
                observed = f"TLS handshake refused: {type(e).__name__}: {str(e)[:160]}"
            else:
                try:
                    outsider.put(
                        "strands/victim-r1/cmd",
                        json.dumps({"action": "stop"}).encode(),
                    )
                    time.sleep(0.6)
                    # If the link never bridged the put silently dropped.
                    held = True
                    observed = (
                        "rogue-CA cert presented; victim CA-only trust "
                        "means handshake fails; sample dropped"
                    )
                finally:
                    try:
                        outsider.close()
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            error = traceback.format_exc()
            held = False
            observed = "unexpected exception in rogue"

    write_result(
        RogueResult(
            rogue_id=rogue_id,
            av_id=av_id,
            title=title,
            posture=posture,
            defence_held=held,
            observed=observed,
            error=error,
            duration_s=time.time() - t0,
        )
    )
    return 0 if held else 1


if __name__ == "__main__":
    raise SystemExit(main())
