"""Repro: run_policy with 100% unresolved keys reports status='error' but the
summary text leads with "Policy complete on '<robot>'" when n_steps < 3.

The early fail-fast at policy_runner.py:2921 raises RuntimeError inside the
3-step probe window and the outer handler composes the text as
"Policy failed: ...". That path reads coherently with status='error'.

The LATE check at policy_runner.py:3283 fires for the SAME failure mode when
n_steps < _FAIL_FAST_PROBE_STEPS (= 3), but it appends to the summary text
built two blocks earlier at policy_runner.py:3099-3106 with
prefix = "Policy complete". The result:

    status: error
    content[0].text: "Policy complete on 'so101'
                      HolosomaPolicy | 
                      0.0s | 1 steps | sim_t=0.000s
                      ...
                      ALL 1 action steps had 100% unresolved keys -- the robot did not move."

A reader (agent or human) scanning the text sees "Policy complete" first and
reads it as success - the exact outcome the fail-fast at line 2921 was built
to prevent (harness-closed #165).

Fix: in the LATE total-failure branches at policy_runner.py:3273 and
policy_runner.py:3311, replace the leading "Policy complete" / "Policy
stopped" prefix with "Policy failed" so the summary header matches the
"Policy failed:" header the EARLY fail-fast uses for the SAME outcome.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strands_robots import Robot


def main() -> int:
    r = Robot("so101", mesh=False)
    bad = 0

    for n in (1, 2):
        res = r.run_policy(policy_provider="holosoma", n_steps=n)
        status = res.get("status")
        text = next((b["text"] for b in res.get("content", []) if "text" in b), "")
        first_line = text.split("\n", 1)[0]
        print(f"n_steps={n}: status={status!r}")
        print(f"  first line: {first_line!r}")
        if status == "error" and first_line.startswith("Policy complete"):
            print(f"  !! status=error but text leads with 'Policy complete' - MISLEADING")
            bad += 1
        print()

    # Control: n_steps=3 hits the early fail-fast; text leads with "Policy failed:"
    res = r.run_policy(policy_provider="holosoma", n_steps=3)
    status = res.get("status")
    text = next((b["text"] for b in res.get("content", []) if "text" in b), "")
    first_line = text.split("\n", 1)[0]
    print(f"n_steps=3 (control, early fail-fast): status={status!r}")
    print(f"  first line (truncated): {first_line[:80]!r}")

    print()
    if bad == 2:
        print(f"REPRODUCED: {bad}/2 short-horizon rollouts reported status=error with")
        print("a text block leading 'Policy complete' - the same outcome the early")
        print("fail-fast at policy_runner.py:2921 raises as 'Policy failed: ...'.")
        return 0  # repro succeeded on buggy main
    print(f"NOT REPRODUCED: {bad}/2 rollouts triggered the misleading header.")
    print("Fix at policy_runner.py:3273,3311 rewrites the leading prefix.")
    return 1  # fix is in place, repro no longer triggers


if __name__ == "__main__":
    sys.exit(main())
