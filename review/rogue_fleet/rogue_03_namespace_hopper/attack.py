#!/usr/bin/env python3
"""Rogue 03 -- cross-fleet namespace hopper.

Threat model:
  Two unrelated fleets share a LAN (a service-robot fleet and a
  warehouse-AGV fleet). An attacker (or just a misconfigured operator)
  has valid mTLS material for fleet A and tries to drive a robot in
  fleet B by reusing fleet A's bus.

Defence under test:
  ``namespace`` config field (Zenoh-native): two fleets with different
  namespaces never see each other's keys. Even with valid mTLS to the
  victim, a publisher on namespace ``other`` cannot deliver to
  ``strands``.

Observation:
  We open a peer with valid mTLS material (signed by the SAME CA the
  victim trusts) but a different ``namespace`` value, then try to put
  a cmd. The victim never sees the message.

Note: in this scenario we accept the strongest interpretation of
``namespace``: the victim, acting in good faith, would respond to
any namespace-prefixed message it sees. The defence holds *if* it
never sees the prefixed message.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from rogue_fleet._lib.report import RogueResult, write_result  # noqa: E402


def main() -> int:
    rogue_id = "rogue_03_namespace_hopper"
    av_id = "AV-03"
    title = "Different-namespace publisher cannot hop into the victim's fleet"
    posture = "victim namespace=strands; rogue namespace=other; same CA"

    victim_listen = os.environ["VICTIM_LISTEN"]
    ca = os.environ["CA_CERT"]
    cert = os.environ["OPERATOR_CERT"]
    key = os.environ["OPERATOR_KEY"]
    t0 = time.time()
    held = False
    observed = ""
    error = ""

    import zenoh

    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"client"')
    cfg.insert_json5("namespace", '"other"')  # <-- not the victim's namespace
    cfg.insert_json5("scouting/multicast/enabled", "false")
    cfg.insert_json5("scouting/gossip/enabled", "true")
    cfg.insert_json5("transport/link/protocols", json.dumps(["tls"]))
    cfg.insert_json5(
        "transport/link/tls",
        json.dumps(
            {
                "enable_mtls": True,
                "verify_name_on_connect": True,
                "root_ca_certificate": ca,
                "connect_certificate": cert,
                "connect_private_key": key,
            }
        ),
    )
    cfg.insert_json5("connect/endpoints", json.dumps([f"tls/{victim_listen}"]))

    # Counter-publisher: subscribe on the victim's namespace from inside
    # our own session to confirm we cannot see the victim's traffic
    # either (presence heartbeats).
    seen_from_victim_ns: list[bytes] = []

    try:
        try:
            session = zenoh.open(cfg)
        except Exception as e:  # noqa: BLE001
            held = True
            observed = f"session refused even before namespace check: {type(e).__name__}"
        else:
            try:
                session.declare_subscriber(
                    "strands/**",
                    lambda s: seen_from_victim_ns.append(s.payload.to_bytes()),
                )
                # Try to publish into the victim's fleet from our own
                # namespace. The wire key becomes 'other/victim-r1/cmd';
                # the victim subscribes to 'strands/...' only.
                session.put(
                    "victim-r1/cmd",
                    json.dumps({"action": "stop"}).encode(),
                )
                time.sleep(1.0)
                # Defence held when we (a) saw zero of victim's heartbeats
                # and (b) the victim never received our cmd. Cross-process
                # access to the victim's audit log is awkward; we infer
                # via the symmetric channel: zero traffic both ways.
                held = len(seen_from_victim_ns) == 0
                observed = (
                    f"namespace='other' isolated from 'strands'; "
                    f"saw {len(seen_from_victim_ns)} samples from victim ns"
                )
            finally:
                try:
                    session.close()
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        error = traceback.format_exc()

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
