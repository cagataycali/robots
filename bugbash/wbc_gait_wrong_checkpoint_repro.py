"""Repro: `wbc_gait` "checkpoint missing" error points to the wrong ONNX family.

Symptom
-------
`sim.run_policy(policy_provider="wbc_gait", ...)` or `create_policy("wbc_gait")`
without a checkpoint raises:

    WBCPolicy requires a checkpoint but none was provided. No weights are
    bundled. Obtain GR00T-WholeBodyControl-Balance.onnx and
    GR00T-WholeBodyControl-Walk.onnx from the NVlabs/GR00T-WholeBodyControl
    git-LFS tree (decoupled_wbc/sim2mujoco/resources/robots/g1/policy/), ...

But `wbc_gait` cannot use those weights. `strands_robots/policies/wbc/gait.py`
(WBCGaitPolicy class docstring, line 309) says:

    "The shipped GR00T-WholeBodyControl-Balance.onnx / -Walk.onnx weights are
     the *non-gait* family (516-wide input); this variant expects a gait-clock
     checkpoint whose ONNX input is [batch, 570] (95 x 6) and output [batch, 15]."

An end-user who follows the error message will download ~200 MB of git-LFS
weights, and then have them rejected at shape-check time.

Root cause
----------
`WBCGaitPolicy` inherits `_checkpoint_not_found_message` from `WBCPolicy`
(strands_robots/policies/wbc/policy.py:1074-1094). The message is hard-coded
for the Balance/Walk pair and is not overridden in the gait subclass, so the
"no checkpoint" path points every wbc_gait user at the wrong ONNX family.

Fix (sketch, ~15 LOC)
---------------------
Override `_checkpoint_not_found_message` in `WBCGaitPolicy` to name the
570-wide gait-clock checkpoint family and NOT the 516-wide Balance/Walk pair,
and to tell the reader that the shipped SONIC weights are the wrong shape for
this provider.

Run
---
    python bugbash/wbc_gait_wrong_checkpoint_repro.py
"""
import textwrap

from strands_robots import Robot


def main() -> int:
    sim = Robot("unitree_g1")
    try:
        sim.run_policy(
            robot_name="unitree_g1",
            policy_provider="wbc_gait",
            duration=0.3,
            control_frequency=10.0,
        )
    except RuntimeError as e:
        msg = str(e)
        print("--- error message shown to end user ---")
        print(textwrap.indent(msg, "    "))
        print()

        # The message names the non-gait Balance/Walk pair:
        assert "GR00T-WholeBodyControl-Balance.onnx" in msg, "missing Balance mention"
        assert "GR00T-WholeBodyControl-Walk.onnx" in msg, "missing Walk mention"
        # It does not mention the 570-wide gait-clock family that wbc_gait needs:
        assert "570" not in msg and "gait" not in msg.lower(), (
            "the error mentions the gait family - did the maintainer fix this?"
        )

        print("PAPERCUT CONFIRMED: the error message directs users to")
        print("  GR00T-WholeBodyControl-Balance.onnx + -Walk.onnx")
        print("but per WBCGaitPolicy's own docstring these are the wrong")
        print("family (516-wide, non-gait). wbc_gait needs [batch, 570].")
        return 1
    print("no error - checkpoint was found, cannot reproduce")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
