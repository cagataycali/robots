#!/usr/bin/env python3
"""Rogue 07 -- ACL shape gate verification (file-load path).

Threat model:
  Operator deploys an ACL file that *looks* valid but silently degrades
  the gate. We exercise the loader (``_load_acl_file``) which is the
  actual operator surface -- not just the shape validator.

Defences:
  1. ``enabled: false`` -> ValueError at load time.
  2. ``interfaces: []`` (empty list) -> ValueError at validate time.
  3. CN-only subject (no ``interfaces`` field) -> accepted (F2 happy path).
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
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rogue_fleet._lib.report import RogueResult, write_result  # noqa: E402


def _write(td: Path, name: str, content: str) -> Path:
    p = td / name
    p.write_text(content)
    return p


def main() -> int:
    rogue_id = "rogue_07_acl_role_violator"
    av_id = "AV-06+07+08"
    title = "ACL loader rejects enabled:false / empty interfaces; accepts CN-only"
    posture = "in-process; ACL file loader under test"

    t0 = time.time()
    held = False
    observed = ""
    error = ""
    outcomes: list[tuple[str, bool]] = []

    try:
        from strands_robots.mesh import _acl_config as ac

        with tempfile.TemporaryDirectory(prefix="acl_") as td:
            tdp = Path(td)

            # 1. enabled: false -- must raise at load
            bad1 = _write(tdp, "bad1.json5", """\
{ enabled: false, default_permission: 'deny', rules: [], subjects: [], policies: [] }
""")
            try:
                ac._load_acl_file(bad1)
                outcomes.append(("enabled_false_raised", False))
            except (ValueError, KeyError):
                outcomes.append(("enabled_false_raised", True))

            # 2. interfaces: [] -- must raise at validate
            bad2 = _write(tdp, "bad2.json5", """\
{
  enabled: true, default_permission: 'deny',
  rules: [{id: 'r0', messages: ['put'], flows: ['ingress'], key_exprs: ['**/cmd'], permission: 'allow'}],
  subjects: [{id: 's0', interfaces: [], cert_common_names: ['operator-1']}],
  policies: [{rules: ['r0'], subjects: ['s0']}]
}
""")
            try:
                ac._load_acl_file(bad2)
                outcomes.append(("empty_interfaces_raised", False))
            except (ValueError, KeyError):
                outcomes.append(("empty_interfaces_raised", True))

            # 3. CN-only -- must NOT raise
            good = _write(tdp, "good.json5", """\
{
  enabled: true, default_permission: 'deny',
  rules: [{id: 'r0', messages: ['put'], flows: ['ingress'], key_exprs: ['**/cmd'], permission: 'allow'}],
  subjects: [{id: 's0', cert_common_names: ['operator-1']}],
  policies: [{rules: ['r0'], subjects: ['s0']}]
}
""")
            try:
                resolved = ac._load_acl_file(good)
                # Round-trip through Config.insert_json5 to verify Zenoh accepts it.
                import zenoh
                cfg = zenoh.Config()
                cfg.insert_json5("access_control", json.dumps(resolved))
                outcomes.append(("cn_only_accepted", True))
            except Exception as e:  # noqa: BLE001
                outcomes.append(("cn_only_accepted", False))
                observed += f" cn_err={type(e).__name__}"

            # 4. Permissive shape detector (F18-B)
            permissive_shape = ac._is_permissive_acl_shape({
                "default_permission": "allow",
                "rules": [], "subjects": [], "policies": [],
            })
            outcomes.append(("permissive_shape_detected", permissive_shape))

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
