#!/usr/bin/env python3
"""Will the arms this dashboard is about to auto-spawn find their calibration?

WHY THIS EXISTS. The dashboard restores arms from ~/.strands_dashboard/profiles.json without being
asked (the USB auto-spawn watcher), so a restart re-creates whatever the profiles say. On this machine
one of them has been wrong since 2026-08-20 and the symptom is invisible where it matters: the peer
appears on the mesh, presence says connected, the camera meta flows, role detection works -- and the
arm reports NO JOINTS AT ALL, because its process died inside lerobot with

    RuntimeError: FeetechMotorsBus(...) has no calibration registered

which is retained for ~10 lines in a per-peer ring buffer and logged at DEBUG by mesh/core. Every
surface a person looks at says the arm is fine. The dashboard cannot record, teleop or train with it.

The mistake is one field. A profile with mode=real and robot_name=so101 is spawned as a ROBOT, so
lerobot resolves calibration/robots/so101_follower/<robot_id>.json; the leader arm's calibration was
made through the TELEOPERATOR path and lives at calibration/teleoperators/so101_leader/leader.json.
Same arm, same file format, a directory apart -- so the file the profile needs is absent while a file
that would answer the question sits one level away. That is precisely the case worth NAMING, because
"no calibration registered" sends you to the arm and the calibration wizard, and the fix is a field.

This check is pure over its inputs (a profiles dict + the set of calibration files that exist), so the
rule is testable without hardware, a filesystem or a running dashboard.

Run:  python3 scripts/check_profile_calibration.py            # reads the real paths
      python3 scripts/check_profile_calibration.py --json     # machine-readable
Exit: 0 = every real-mode profile has its calibration; 1 = at least one cannot spawn; 2 = nothing to
check (no profiles found), which is NOT a pass -- see the narrowing law below.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# lerobot's own layout: <root>/robots/<class>/<id>.json for a robot,
# <root>/teleoperators/<class>/<id>.json for a teleoperator.
CALIB_ROOT = Path.home() / ".cache/huggingface/lerobot/calibration"
PROFILES = Path.home() / ".strands_dashboard/profiles.json"

# robot_name as the dashboard stores it -> (robot class dir, teleoperator class dir).
# Only the families this dashboard spawns; an unknown family is REPORTED, never guessed at.
FAMILIES = {
    "so101": ("so101_follower", "so101_leader"),
    "so100": ("so100_follower", "so100_leader"),
}


def calibration_verdicts(profiles: dict, existing: set[str]) -> list[dict]:
    """One verdict per real-mode profile. `existing` holds paths relative to the calibration root.

    Pure: no filesystem, no network, no hardware.
    """
    out: list[dict] = []
    for serial, p in sorted((profiles or {}).items()):
        if not isinstance(p, dict) or p.get("mode") != "real":
            continue  # a sim twin needs no calibration; say nothing about it
        peer = p.get("peer_id") or p.get("name") or serial
        family = str(p.get("robot_name") or "")
        robot_id = str(p.get("robot_id") or "")
        fam = FAMILIES.get(family)
        if fam is None:
            out.append({
                "peer": peer, "serial": serial, "ok": False, "reason": "unknown_family",
                "detail": f"robot_name {family!r} is not a family this check knows, so it cannot say "
                          f"where {peer}'s calibration should live. Add it to FAMILIES.",
            })
            continue
        robot_dir, teleop_dir = fam
        wanted = f"robots/{robot_dir}/{robot_id}.json"
        if wanted in existing:
            out.append({"peer": peer, "serial": serial, "ok": True, "path": wanted})
            continue

        # The interesting case: the same id calibrated through the teleoperator path.
        as_teleop = f"teleoperators/{teleop_dir}/{robot_id}.json"
        siblings = sorted(e for e in existing if e.startswith(f"robots/{robot_dir}/"))
        if as_teleop in existing:
            detail = (
                f"{peer} is spawned as a ROBOT (mode=real, robot_name={family}) so lerobot needs "
                f"{wanted}, which does not exist -- but this arm IS calibrated, as a TELEOPERATOR, at "
                f"{as_teleop}. Same arm, same format, a directory apart. Either copy that file to "
                f"{wanted}, or give the profile a robot_id whose robot-side calibration exists "
                f"({', '.join(Path(s).stem for s in siblings) or 'none'})."
            )
            reason = "calibrated_as_teleoperator"
        else:
            detail = (
                f"{peer} needs {wanted} and no calibration answers to that id. Robot-side ids that DO "
                f"exist for this family: {', '.join(Path(s).stem for s in siblings) or 'none'}. "
                f"Calibrate this arm, or point the profile at one of those."
            )
            reason = "missing"
        out.append({
            "peer": peer, "serial": serial, "ok": False, "reason": reason,
            "wanted": wanted, "detail": detail,
        })
    return out


def _existing(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {str(p.relative_to(root)) for p in root.rglob("*.json")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", default=str(PROFILES))
    ap.add_argument("--calibration-root", default=str(CALIB_ROOT))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    prof_path, root = Path(args.profiles), Path(args.calibration_root)
    try:
        profiles = json.loads(prof_path.read_text())
    except FileNotFoundError:
        print(f"  no profiles at {prof_path} — nothing will auto-spawn, so nothing is checked")
        return 2
    except json.JSONDecodeError as e:
        print(f"  FAIL  {prof_path} is not valid JSON ({e}); the dashboard cannot read it either")
        return 1

    existing = _existing(root)
    verdicts = calibration_verdicts(profiles, existing)

    if args.json:
        print(json.dumps({"checked": len(verdicts), "profiles": len(profiles),
                          "calibration_files": len(existing), "verdicts": verdicts}, indent=2))
    else:
        # THE NARROWING LAW (BUGS.md): a tool that can be narrowed must report X of Y, and must not
        # exit 0 when narrowed-and-empty. "0 problems" over 0 arms is the false green that fooled an
        # agent auditing the audits.
        print(f"  {len(verdicts)} real-mode arm(s) checked of {len(profiles)} profile(s); "
              f"{len(existing)} calibration file(s) on disk")
        for v in verdicts:
            if v["ok"]:
                print(f"  ok      {v['peer']} -> {v['path']}")
            else:
                print(f"  FAIL    {v['peer']}: {v['detail']}")

    if not verdicts:
        print("  nothing to check: no profile has mode=real, which is not the same as 'all good'")
        return 2
    if not existing:
        print(f"  no calibration files under {root} — this check is looking in the wrong place, which")
        print("  is worse than the bug it guards against")
        return 2
    return 1 if any(not v["ok"] for v in verdicts) else 0


if __name__ == "__main__":
    sys.exit(main())
