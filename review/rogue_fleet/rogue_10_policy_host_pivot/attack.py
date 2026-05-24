#!/usr/bin/env python3
"""Rogue 10 -- policy-host CIDR allowlist + IPv6 server_address parser.

Threat model:
  Two related attacks against the inference-pivot defence:

  1. Bad CIDR / hostname inputs to ``is_safe_policy_host`` should
     fail closed, never throw, and never accept attacker-controlled
     hostnames just because the env var is malformed.
  2. The ``server_address`` parser used by the agent runtime must
     handle IPv6 forms (``[::1]:8000`` etc.) without confusing the
     port boundary.

Defences:
  * ``security.is_safe_policy_host`` -- loopback default; explicit
    CIDR / host allowlist via ``STRANDS_MESH_POLICY_HOST_ALLOW``.
  * ``security.is_safe_server_address`` -- composite host[:port]
    parser that handles IPv6 + IPv4 + DNS forms.
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
    rogue_id = "rogue_10_policy_host_pivot"
    av_id = "AV-10+33"
    title = "is_safe_policy_host and is_safe_server_address parse correctly"
    posture = "in-process; CIDR + hostname + IPv6 inputs"

    t0 = time.time()
    held = False
    observed = ""
    error = ""
    outcomes: list[tuple[str, bool]] = []

    try:
        from strands_robots.mesh import security as sec

        # Loopback always OK
        outcomes.append(("loopback_allowed", sec.is_safe_policy_host("127.0.0.1")))
        outcomes.append(("localhost_allowed", sec.is_safe_policy_host("localhost")))
        # Public host disallowed by default
        outcomes.append(("public_blocked", not sec.is_safe_policy_host("evil.attacker.com")))
        # Operator extends via env
        os.environ["STRANDS_MESH_POLICY_HOST_ALLOW"] = "vla.internal,10.0.0.0/24"
        outcomes.append(("explicit_host_allowed", sec.is_safe_policy_host("vla.internal")))
        outcomes.append(("cidr_member_allowed", sec.is_safe_policy_host("10.0.0.7")))
        outcomes.append(("cidr_outside_blocked", not sec.is_safe_policy_host("10.1.0.7")))
        # Malformed env entries dropped (fail-loud, not fail-open)
        os.environ["STRANDS_MESH_POLICY_HOST_ALLOW"] = "vla.internal,;rm -rf /"
        outcomes.append(("malformed_dropped", sec.is_safe_policy_host("vla.internal")))
        outcomes.append(("injection_not_accepted",
                         not sec.is_safe_policy_host(";rm -rf /")))
        os.environ.pop("STRANDS_MESH_POLICY_HOST_ALLOW", None)

        # IPv6 server_address parsing
        if hasattr(sec, "is_safe_server_address"):
            outcomes.append(("ipv6_loopback", sec.is_safe_server_address("[::1]:8000")))
            outcomes.append(("ipv4_loopback", sec.is_safe_server_address("127.0.0.1:8000")))
            outcomes.append(("hostname_form", sec.is_safe_server_address("localhost:8000")))
            outcomes.append(("public_v6_blocked",
                             not sec.is_safe_server_address("[2001:db8::1]:8000")))
            outcomes.append(("missing_port_allowed",  # bare host is OK, just checks host allowlist
                             sec.is_safe_server_address("127.0.0.1")))
            # F18 fix: ``[::1`` (unmatched bracket) must be rejected
            outcomes.append(("unmatched_bracket_blocked",
                             not sec.is_safe_server_address("[::1:8000")))

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
