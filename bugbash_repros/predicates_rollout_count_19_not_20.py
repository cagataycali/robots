"""Repro + pin for the docs/code mismatch in predicates-and-rollouts.md.

BEFORE this branch, docs/learn/simulation/predicates-and-rollouts.md:51
promised the "You should see" block ended with::

    RunPolicyStarted RunPolicyEnded predicate 20

A fresh user who copy-pastes the deterministic sketch above it ran it and
saw::

    RunPolicyStarted RunPolicyEnded predicate 19

The 20 was wrong: so101 + MockPolicy sinusoid + mujoco default physics at
50 Hz is deterministic, and `joint_above("1", 0.3)` fires on event 19.

This script reproduces the deterministic value and pins it, so a future
rewrite of the sketch does not drift from the docs again.

Run::

    MUJOCO_GL=egl python predicates_rollout_count_19_not_20.py

Exit code 0 on the pinned 19, 1 on anything else (including 20).
"""
from __future__ import annotations

import sys

from strands_robots.simulation import create_simulation


def main() -> int:
    sim = create_simulation("mujoco")
    sim.create_world()
    sim.add_robot("so101")
    sim.add_object(
        name="cube",
        shape="box",
        size=[0.03, 0.03, 0.03],
        position=[0.25, 0.0, 0.015],
    )

    events: list = []
    sim.run_policy(
        robot_name="so101",
        policy_provider="mock",
        n_steps=200,
        control_frequency=50.0,
        stop_when={"predicate": "joint_above", "joint": "1", "value": 0.3},
        observer=events.append,
    )

    reason = events[-1].stopped_reason
    applied = events[-1].applied_actions
    print(f"stopped_reason={reason!r} applied_actions={applied}")

    if reason != "predicate":
        print(
            f"[FAIL] stopped_reason must be 'predicate', got {reason!r}",
            file=sys.stderr,
        )
        return 1
    if applied != 19:
        print(
            f"[FAIL] applied_actions must be 19 (deterministic), got {applied}. "
            "If this changed, update docs/learn/simulation/predicates-and-rollouts.md "
            "line 58 to match.",
            file=sys.stderr,
        )
        return 1
    print("[PASS] docs expected value matches actual.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
