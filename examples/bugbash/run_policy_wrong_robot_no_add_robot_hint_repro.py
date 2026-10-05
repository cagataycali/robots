#!/usr/bin/env python3
"""Minimal repro: _unknown_robot_msg suggests the loaded sibling as if the
caller mistyped, when the caller named a different registered robot.

Expected pre-fix:
    Robot('so100') not found. Did you mean: so101? Available robots: ['so101'].

That is misleading in two ways:
    1. 'so100' is NOT a typo of 'so101' - both are registered robots.
    2. "Available robots: ['so101']" implies so101 is the only robot that
       exists in the ecosystem (there are 157 registered).

The only recovery is add_robot('so100'), but the error never names it.

Expected post-fix:
    Robot 'so100' not found. 'so100' is a registered robot but is not loaded
    in this scene; add it first with action='add_robot', name='so100'. Robots
    in the scene: ['so101'].
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"

from strands_robots import Robot, list_robots

so100_in_registry = any(r["name"] == "so100" for r in list_robots())
assert so100_in_registry, "so100 should be in the registry"

r = Robot("so101", mesh=False)
result = r.run_policy("so100", policy_provider="mock", instruction="walk", duration=0.1)

text = result["content"][0]["text"]
print("ACTUAL ERROR:")
print(" ", text)
print()

# Pre-fix: these two assertions pass. Post-fix: both flip.
BAD_HINT = "Did you mean: so101?"
BAD_LISTING = "Available robots: ['so101']."

if BAD_HINT in text:
    print(f"FAIL: error carries misleading 'did you mean' hint — {BAD_HINT!r}")
    print("      so100 is not a typo of so101; both are registered.")
    sys.exit(1)
if BAD_LISTING in text:
    print(f"FAIL: error claims only so101 is available (there are 157 registered robots)")
    sys.exit(1)
if "add_robot" not in text:
    print(f"FAIL: error does not mention the actual recovery (add_robot)")
    sys.exit(1)

print("PASS: error is actionable (names add_robot, does not falsely suggest typo)")
