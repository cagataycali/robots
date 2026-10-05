"""Repro: `send_action({})` reports status='success' with the message
    "Action applied to '<robot>' (0 keys)." - *and* advances physics
    ``n_substeps`` times. The two words together ("applied" and "0 keys")
    tell an LLM reading the envelope that nothing happened, when the world
    in fact advanced ``n_substeps`` physics substep(s).

The *behavioral* question (refuse empty mapping entirely vs. silently step)
is pinned to the current shape by TWO tests:

  - tests/simulation/isaac/test_delta_eef_controller.py (empty-dict-via-
    controller is a documented "clean settle" idiom).
  - tests/simulation/mujoco/test_a_diverged_world_is_reported_not_stepped_through.py
    (empty-dict is the probe used to drive the divergence-report path on
    the raw send_action; it must still step 5 substeps to find the diverged
    world).

Both pin "empty dict ADVANCES physics" as the design intent, so the fix on
this branch is the SMALLEST honest answer: change the success text so the
two readings ("no actuators commanded" and "world advanced n substeps") are
visible to the caller. The behavioural question is left as a B-tier
follow-up for the maintainers.

Compare to the sibling guards (unchanged):
  * send_action([])    -> status='error' ("length 0 does not match ...")
  * send_action("x")   -> status='error' ("'action' must be a mapping ...")
  * add_object(name="")-> status='error' ("'name' must be a non-empty string, ...")

Only the empty-dict path reports success - which is the pinned shape, so
the message is what gets fixed.

Upstream (v0.5.3, commit 496ee95c0):
  strands_robots/simulation/mujoco/simulation.py:1374  "Action applied ..."
  strands_robots/simulation/newton/simulation.py:1387  same wording
  strands_robots/simulation/isaac/simulation.py:6704   same wording
  strands_robots/simulation/base.py:_coerce_action     the mapping branch

Run:
    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install 'strands-robots[sim-mujoco]'
    python bugbash_repros/send_action_empty_mapping_silent_step_repro.py
"""
from __future__ import annotations

import sys


def main() -> int:
    from strands_robots import Robot

    robot = Robot("lekiwi", position=[0.0, 0.0, 0.0346])
    try:
        t0 = robot.mj_data.time
        result = robot.send_action({})
        t1 = robot.mj_data.time

        status = result.get("status")
        text = result["content"][0]["text"]
        advanced = t1 - t0

        print(f"send_action({{}}) status  = {status!r}")
        print(f"send_action({{}}) message = {text!r}")
        print(f"world.time       delta   = {advanced:.6f} s")
        print()

        print("Sibling guards for reference:")
        for action in ([], (), "hello", None):
            r = robot.send_action(action)
            print(f"  send_action({action!r:20}) -> status={r.get('status')!r}")

        # Defect manifests when the success text says "applied" with a zero
        # count AND the world advanced. On the fixed branch the text is
        # rewritten to name the two things separately.
        says_applied = "applied" in text.lower() and "0 keys" in text
        if status == "success" and advanced > 0 and says_applied:
            print()
            print("DEFECT: success text reads 'applied (0 keys)' while world "
                  f"advanced {advanced*1000:.1f} ms.")
            print("Fix: name 'no actuators commanded' and 'advanced n substeps' separately.")
            return 2
        print()
        print("OK: envelope separates the two readings.")
        return 0
    finally:
        robot.cleanup()


if __name__ == "__main__":
    sys.exit(main())
