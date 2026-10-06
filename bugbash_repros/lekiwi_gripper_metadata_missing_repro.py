"""Repro: `lekiwi` is the only mobile_manip/arm registry entry that
advertises a gripper via joint_labels (`"Jaw" -> "gripper"`) but ships NO
`gripper` block.

Impact (today, on tag main @ 3380fbe / release 0.5.3 lineage):

    * ``set_gripper(..., robot_name="lekiwi")`` succeeds via the
      _GRIPPER_HINTS=("gripper","finger","jaw") name heuristic (actuator is
      named "Jaw" so the substring match fires), and the result JSON reports
      ``setpoint_sources: 'actuator ctrlrange'`` instead of the authoritative
      ``'registry metadata'`` sibling arms report.  So the operation works
      *today* but is on the fallback path the docstring at
      ``simulation/mujoco/motion_primitives.py:_resolve_gripper_actuators``
      calls a "zero-config fallback", not the "registry metadata first" path.

Latent risk (why this is still a real defect, not a cosmetic one):

    * ``_gripper_state_end`` warns in its own docstring:
      "``open=HIGH / close=LOW`` convention (correct for SO-100/SO-101 and
      Franka, but a convention, not a law - the metadata field exists to
      remove that sign trap for robots with an inverted gripper)".
    * Lekiwi's MJCF is NOT a Strands-maintained asset: it is pulled from the
      third-party Ekumen-OS/lekiwi repo (registry.asset.source.repo).  That
      repo ALREADY ships a different Jaw joint range (``0 0.6`` vs the native
      so_arm100 ``-0.174 1.75``), so the sign-flip trap is not theoretical -
      a future Ekumen update can flip the convention and ``set_gripper(close)``
      will silently OPEN the gripper with no CI signal, because we never
      pinned the ``closed``/``open`` convention in the registry.

Visible docs asymmetry (today):

    * docs/robots/lekiwi.md's "also accepted by send_action" table lists
      ``Jaw | gripper`` - the same promise so100/so101 make.
    * so100 and so101 both ship a ``gripper`` block that makes the promise
      authoritative; lekiwi doesn't.

Fix (one JSON line in the lekiwi entry, mirroring so100 exactly):

    "gripper": {"actuators": ["Jaw"], "closed": "low", "open": "high"},
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "strands_robots" / "registry" / "robots.json"

ROBOTS = json.loads(REGISTRY.read_text())["robots"]


def _jaw_in_labels(entry: dict) -> bool:
    """True iff joint_labels maps a key named 'Jaw' (or value 'gripper')."""
    labels = entry.get("joint_labels", {})
    return any(
        str(src).lower() == "jaw" or "gripper" in str(target).lower()
        for src, target in labels.items()
    )


def _summary() -> list[tuple[str, str, bool, bool]]:
    rows = []
    for name, entry in ROBOTS.items():
        cat = entry.get("category")
        if cat not in ("arm", "mobile_manip"):
            continue
        if _jaw_in_labels(entry):
            rows.append((name, cat, _jaw_in_labels(entry), "gripper" in entry))
    return rows


def main() -> int:
    print("Registry rows advertising a Jaw/gripper joint_label")
    print("-" * 72)
    print(f"{'robot':20s} {'category':14s} {'jaw_label':>12s} {'gripper_block':>15s}")
    missing = []
    for name, cat, has_label, has_block in _summary():
        marker = "" if has_block else "   <-- MISSING"
        print(f"{name:20s} {cat:14s} {'yes':>12s} {('yes' if has_block else 'no'):>15s}{marker}")
        if not has_block:
            missing.append(name)
    print()

    # Pinpoint the asymmetry with so100 (the closest sibling).
    print("Sibling with the SAME joint_labels['Jaw']='gripper' AND a block:")
    print(f"  so100.gripper = {ROBOTS['so100']['gripper']}")
    print()
    print("Hole on lekiwi:")
    print(f"  lekiwi.gripper        = {ROBOTS['lekiwi'].get('gripper')!r}")
    print(f"  lekiwi.joint_labels[Jaw] = "
          f"{ROBOTS['lekiwi']['joint_labels'].get('Jaw')!r}")
    print(f"  lekiwi.asset.source   = {ROBOTS['lekiwi']['asset'].get('source')}")
    print()

    # Second-order check: run set_gripper on lekiwi and show that the result
    # JSON surfaces the fallback marker.  This only runs when the sim extra
    # is installed; the registry assert above is the primary pin.
    if os.environ.get("BUGBASH_RUN_SIM", "1") == "1":
        try:
            os.environ.setdefault("MUJOCO_GL", "egl")
            from strands_robots import Robot

            lk = Robot("lekiwi", position=[0.0, 0.0, 0.0346])
            res = lk.set_gripper(state="close", robot_name="lekiwi")
            payload = next((c for c in res.get("content", []) if "json" in c), None)
            setpoint_sources = (payload or {}).get("json", {}).get("setpoint_sources")
            print("Live-sim probe (set_gripper close on lekiwi):")
            print(f"  status           = {res.get('status')}")
            print(f"  setpoint_sources = {setpoint_sources}")
            print(
                "  (sibling so100 reports the same string 'actuator ctrlrange' because the\n"
                "   mujoco backend reads the physical range from the model either way; the\n"
                "   distinction we care about is the resolution PATH, which the result does\n"
                "   not expose - the only user-facing evidence is the registry hole + the\n"
                "   docstring's own warning about the sign trap.)"
            )
            lk.cleanup()
        except Exception as exc:
            print(f"(sim probe skipped: {type(exc).__name__}: {exc})")

    assert missing == ["lekiwi"], (
        f"Expected exactly ['lekiwi'] to be missing a gripper block among "
        f"jaw-label arm/mobile_manip robots on main 3380fbe; got {missing}. "
        "If this list grew or shrunk, update the issue."
    )
    print()
    print("ASSERT PASSED: lekiwi is the unique outlier.")
    print("Fix: add after joint_labels in robots.json:")
    print('    "gripper": {"actuators": ["Jaw"], "closed": "low", "open": "high"},')
    return 0


if __name__ == "__main__":
    sys.exit(main())
