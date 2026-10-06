"""Repro for strands-labs/robots v0.5.3 bugbash defect: `docs/learn/hardware/feetech-arms.md:18,24`
claims the default driver for `Robot("so101", mode="real", ...)` is `lerobot`.

The actual default is `strands` (the native FeetechDriver), because
`strands_robots.drivers.registry.resolve_driver(canonical, None)` returns
``NATIVE_DRIVER`` whenever a native driver is registered for the canonical
name (:func:`get_native_driver_class`) - and ``so100``/``so101``/``lekiwi``/
``hope_jr``/``open_duck_mini``/``koch``/``ur5e``/``g1``/``booster_t1``/
``microduck`` all have one.

The paper-cut: a user reading the quickstart and the comparison table (line 24,
'driver="lerobot" (default)') expects lerobot semantics (lerobot calibration
file, `shoulder_pan.pos` units, cameras opened by lerobot, `use_degrees=True`,
`gripper.pos` 0-100). They actually get FeetechDriver semantics (strands
calibration via `calibration=`, `shoulder_pan` or `shoulder_pan.pos` keys,
cameras not read, percent-open gripper). The behavioural delta is the entire
"Which driver" table on the same page.

Run:
    cd .../robots
    python bugbash_repros/feetech_arms_md_default_driver_wrong_repro.py

Expected (per docs): resolved driver == 'lerobot' for the default call.
Actual: resolved driver == 'strands'.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from strands_robots.drivers.registry import resolve_driver


def main() -> int:
    # The call shape at docs/learn/hardware/feetech-arms.md:18
    # `arm = Robot("so101", mode="real", port="/dev/ttyACM0")  # lerobot driver, the default`
    # The driver chosen by the factory for this call is governed by
    # resolve_driver(canonical, explicit=None). Call it directly (no bus needed).
    rows = []
    for canonical in [
        "so101",            # explicitly named in docs/learn/hardware/feetech-arms.md:18
        "so100",            # same table says "both register for so100 ... (line 34)
        "lekiwi",           # idem
        "hope_jr",          # idem
        "open_duck_mini",   # idem
    ]:
        got = resolve_driver(canonical, None)   # None mirrors a missing `driver=` kwarg
        rows.append((canonical, got))

    docs_claim = "lerobot"  # per feetech-arms.md:18 inline comment and :24 table header
    width = max(len(r[0]) for r in rows)

    print("docs/learn/hardware/feetech-arms.md:18 says: default driver is 'lerobot'.")
    print("docs/learn/hardware/feetech-arms.md:24 table header: driver=\"lerobot\" (default).")
    print()
    print("Resolved by strands_robots.drivers.registry.resolve_driver(canonical, None):")
    print()
    fails = 0
    for canonical, got in rows:
        tag = "OK" if got == docs_claim else "DOCS WRONG"
        if got != docs_claim:
            fails += 1
        print(f"  {canonical:<{width}}  default -> {got!r:<12}  [{tag}]")

    print()
    if fails:
        print(f"FAIL: {fails}/{len(rows)} robots resolve to a driver other than the docs' claim.")
        print("Source of truth: strands_robots/drivers/registry.py:65-108")
        print("(a canonical with a native driver returns NATIVE_DRIVER='strands';")
        print(" DEFAULT_DRIVER='lerobot' is reached ONLY when get_native_driver_class is None).")
        return 1
    print("OK: default driver matches docs for every tested robot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
