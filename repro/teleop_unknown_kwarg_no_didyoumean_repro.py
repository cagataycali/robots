"""Repro: Teleoperator()'s unknown-kwarg refusal is the only one in the family
that doesn't offer a "Did you mean" hint — a typo like ``prot`` instead of
``port`` dumps 40+ allowlist entries at the user instead of a one-line arrow.

Three siblings in the same package offer the courtesy for the same shape of
mistake:

- strands_robots/robot.py:211         Robot("so1000") -> "Did you mean: so100, so101?"
- strands_robots/policies/factory.py:398,648
                                      create_policy("wbc_gaid") -> same shape
- strands_robots/hardware_robot.py:252,302
                                      Robot(cameras={type: 'opencv', index_or_pat: 0})
                                      -> "'index_or_pat' -> 'index_or_path'"

Only strands_robots/teleoperator.py:203 is missing it, so a two-letter typo
spits out a wall of 44 names and the user has to visual-diff them.

Run with:  python repro/teleop_unknown_kwarg_no_didyoumean_repro.py
Expected (fixed): refusal message includes "Did you mean: 'prot' -> 'port'?"
Actual (today):   refusal message dumps 44 field names, no arrow.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("STRANDS_MESH", "false")

    from strands_robots import Teleoperator

    # The classic single-key typo: "prot" for "port". Any leader with a serial
    # bus suffices; so101_leader is a cross-platform common one.
    try:
        Teleoperator("so101_leader", prot="/dev/ttyACM0")
    except ValueError as exc:
        msg = str(exc)
        print("teleoperator refusal:", msg)
        has_hint = "Did you mean" in msg and "'port'" in msg
        print()
        print("sibling (hardware_robot._build_camera_config) does the courtesy:")
        from strands_robots import Robot

        try:
            Robot(
                "so101",
                mode="real",
                driver="lerobot",
                cameras={"front": {"type": "opencv", "index_or_pat": 0}},
                port="/dev/ttyACM0",
            )
        except ValueError as sibling_exc:
            print("camera refusal:   ", str(sibling_exc))

        print()
        if not has_hint:
            print("FAIL: teleop refusal has no 'Did you mean' hint "
                  "while the camera sibling does.")
            return 1
        print("OK: teleop refusal gained the 'Did you mean' hint.")
        return 0
    print("FAIL: expected ValueError on unknown kwarg; none raised.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
