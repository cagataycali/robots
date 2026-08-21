#!/usr/bin/env python
"""Which FIELDS has the running dashboard never heard of? (the blind spot of audit-server-age)

`audit-server-age.mjs` compares PATHS, so a route that exists but has grown a new response field looks
identical to a current one. That gap is not hypothetical - it has now bitten twice on this machine:

  * `/api/config` grew `security.notice` (the token-readable-via-`ps` warning) - route unchanged;
  * `/api/training/datasets` grew `usable`/`problem`, so the running server offered an abandoned
    session's empty folder as a normal training target while the source knew better.

Both were found by hand. This finds them by measurement: build the app from THIS SOURCE in-process
(no port, no restart, and `STRANDS_MESH=false` - the documented hard kill switch, verified in Q32 to
stop `_gateway_mesh` joining the live fleet), then GET the same read-only routes from the RUNNING
server and diff the field names one nesting level deep.

Read-only by construction: only GET, and only routes that report state. It never spawns, never writes
and never touches a motor.

    .venv/bin/python scripts/audit_server_fields.py          # exits 0; NEWS lines are the point
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE = os.environ.get("DASH", "http://127.0.0.1:8090")
TOKEN_FILE = Path.home() / ".strands_dashboard" / "local_api_token.txt"

#: Routes whose fields the UI reasons about. Each must be a GET that only reports.
ROUTES = (
    "/api/config",
    "/api/health",
    "/api/fleet",
    "/api/training/datasets?hub=false",
    "/api/training/jobs",
    "/api/training/trainers",
    "/api/policies",
    "/api/devices",
    "/api/devices/profiles",
    "/api/devices/arm-role",
    "/api/calibration",
    "/api/consent",
    "/api/agent/status",
    "/api/auth/status",
    "/api/mesh/config",
    "/api/robots/registry",
    "/api/record/session",
    "/api/activity",
)

#: WHAT THIS CANNOT SEE, stated where the reader is: a field the source only emits under a CONDITION
#: is invisible unless the condition also holds in-process. `/api/config`'s `security.notice` is the
#: worked example - it appears only when the process was started with `--auth-token` in argv, which a
#: TestClient app never is, so this audit reports nothing about it and is right not to guess. A field
#: this audit does not name is therefore "not observed", never "absent from the source".


def field_names(value: Any, prefix: str = "", depth: int = 2) -> set[str]:
    """Field paths one or two levels deep, with lists represented by their FIRST element.

    A list is sampled, not merged: rows come from one code path, so the first row's keys are that
    path's shape. Merging every row would hide a field that only some rows carry, which is the
    opposite of what this audit is for.
    """
    out: set[str] = set()
    if depth <= 0:
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            path = f"{prefix}{k}"
            out.add(path)
            out |= field_names(v, f"{path}.", depth - 1)
    elif isinstance(value, list) and value:
        out |= field_names(value[0], f"{prefix}[].", depth)
    return out


def live(path: str, token: str) -> tuple[int, Any]:
    req = urllib.request.Request(f"{BASE}{path}", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 - fixed localhost base
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:  # noqa: BLE001 - no server is a SKIP, not a failure
        print(f"  SKIP  no running server at {BASE}: {e}")
        return 0, None


def main() -> int:
    if not TOKEN_FILE.exists():
        print(f"  SKIP  no api token at {TOKEN_FILE}")
        return 0
    token = TOKEN_FILE.read_text().strip()

    os.environ["STRANDS_MESH"] = "false"  # BEFORE the import: nothing may join the live fleet
    # /api/devices ENUMERATES hardware (serial ports, and camera indices through OpenCV). Harmless -
    # it opens nothing the live dashboard holds and every open fails outright in a daemon-descended
    # process with no camera grant - but without this the run prints a wall of AVFoundation
    # authorisation noise around the actual findings.
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
    from fastapi.testclient import TestClient

    from strands_robots.dashboard.server import create_app

    client = TestClient(create_app())

    checked = 0
    news = 0
    for path in ROUTES:
        status, live_body = live(path, token)
        if status == 0:
            return 0  # server gone: the whole comparison is meaningless, say nothing else
        src = client.get(path, headers={"Authorization": f"Bearer {token}"})
        if status != 200 or src.status_code != 200:
            print(f"  note  {path}: live {status}, source {src.status_code} — not comparable")
            continue
        checked += 1
        src_fields = field_names(src.json())
        live_fields = field_names(live_body)
        dark = sorted(src_fields - live_fields)
        gone = sorted(live_fields - src_fields)
        if dark:
            news += 1
            print(f"  NEWS  {path}: {len(dark)} field(s) this SOURCE sends that the RUNNING server does not:")
            for f in dark:
                print(f"  note    dark field → {f}")
        if gone:
            # ASYMMETRIC ON PURPOSE. The dark side is news; this side is mostly DATA, not code: the
            # in-process app has no mesh (kill switch), no peers and no live session, so keys that are
            # per-peer or per-port names ("peers.so101-follower", "mesh.port") are absent because there
            # is nothing to report, not because the source dropped them. Printed as one line, never as
            # a finding, so nobody spends an iteration chasing a field the audit itself removed.
            print(f"  note  {path}: {len(gone)} field(s) only the running server sends"
                  f" (usually live DATA this in-process app has none of): {', '.join(gone[:6])}")
        if not dark and not gone:
            print(f"  ok    {path}: same fields")

    # The narrowing disclosure this repo requires: an empty comparison is not a clean bill of health.
    print(f"  {'NEWS' if news else 'PASS'}  {checked} of {len(ROUTES)} routes compared"
          f"{'' if checked == len(ROUTES) else ' (the rest were not comparable — see the notes)'}")
    if not checked:
        print("  FAIL  nothing was actually compared — treat this as no information, not as agreement")
        return 1
    print("        → dark fields are features waiting on an owner-run restart from a terminal;")
    print("          RESTART_NOTES.md is where they get written down.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
