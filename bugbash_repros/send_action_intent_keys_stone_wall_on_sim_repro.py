"""
Repro: `send_action({"vx": 0.1, "vyaw": 0.0})` on a sim microduck/g1 is refused
with no hint that the SAME keys ARE the canonical intent on `mode="real"`,
and no pointer to the sim-equivalent twist path (`run_policy` with
`policy_provider="microduck"` and `policy_kwargs={"target_velocity": [...]}`).

End-user path (docs/learn/hardware/microduck.md:17 teaches):

    duck = Robot("microduck", mode="real")
    duck.send_action({"vx": 0.1, "vyaw": 0.0})       # walks

User flips to sim (common copy-paste; same `Robot("microduck")` constructor
docs/robots/microduck.md:16 teaches):

    sim = Robot("microduck")                          # sim
    sim.send_action({"vx": 0.1, "vyaw": 0.0})         # stone wall

Refusal (strands_robots/simulation/mujoco/simulation.py:1397-1413,
`_unresolved_action_refusal`):

    Keys ['vx', 'vyaw'] could not be resolved to actuators or joints on
    'microduck'. Nothing was applied and the world did not advance.
    Use individual joint/actuator names as dict keys. Valid keys: [...14
    joint names...]

The hint lists 14 joint names and nothing else:

- Does NOT name that `vx`/`vy`/`vyaw` ARE legitimate intents on `mode="real"`
  for this robot (same docs page the user came from).
- Does NOT point at the sim-equivalent twist path
  `run_policy(robot_name="microduck", policy_provider="microduck",
             policy_kwargs={"target_velocity": [vx, vy, vyaw]})` (which the
  shipped `alpha_walking.onnx` reads 1:1).
- Hits every intent-level driver family identically (g1 and microduck both
  silently stone-wall the same way; see the `g1` check below).

Class: Error UX + Sim/Real Asymmetric API + Docs mismatch.

Fix sketch (not applied on this branch; left for maintainers): in
`_unresolved_action_refusal`, when ALL `unresolved` keys are known intent
keys of an intent-level driver on this robot family (microduck's
`_ACTION_KEYS` at strands_robots/drivers/microduck.py:256-268, or
booster's twist signature), add one line naming the real-mode intent path
AND the sim-equivalent `run_policy(target_velocity=...)`.
"""

from __future__ import annotations

import sys
from strands_robots import Robot


# ---------- microduck -------------------------------------------------------
sim = Robot("microduck")

assert "vx" not in sim.robot_action_keys("microduck"), (
    "sim microduck does not expose intent keys as joints"
)

result = sim.send_action({"vx": 0.1, "vyaw": 0.0})
sim.cleanup()

text = result["content"][0]["text"]
assert result["status"] == "error", result
assert "could not be resolved" in text, text

# The refusal makes NONE of the three following improvements:
assert "mode=" not in text and "mode='real'" not in text, (
    "unexpected: refusal already names mode='real' -- the paper-cut is partly fixed"
)
assert "run_policy" not in text, (
    "unexpected: refusal already points at the sim-equivalent run_policy path"
)
assert "intent" not in text, (
    "unexpected: refusal already names intent keys"
)
assert "target_velocity" not in text, (
    "unexpected: refusal already names policy_kwargs=target_velocity"
)

print("microduck sim refusal (truncated):")
print(" ", text[:200], "...")
print()

# ---------- g1 (same intent-driver family) ----------------------------------
# The G1 is an intent-level robot via `BoosterDriver`
# (strands_robots/drivers/booster.py:851 -- move(vx, vy, vyaw)); the sim refusal
# gives the same stone wall:
g1_sim = Robot("g1")
res_g1 = g1_sim.send_action({"vx": 0.1, "vyaw": 0.0})
g1_sim.cleanup()
text_g1 = res_g1["content"][0]["text"]
assert res_g1["status"] == "error", res_g1
assert "could not be resolved" in text_g1, text_g1
assert "run_policy" not in text_g1 and "mode=" not in text_g1, text_g1

print("g1 sim refusal (truncated):")
print(" ", text_g1[:200], "...")
print()

# ---------- sanity: the sim-equivalent twist path IS run_policy -------------
# The ONNX export `alpha_walking.onnx` reads a 3-vector `target_velocity`
# (strands_robots/policies/microduck/policy.py); the shipped sketch at
# docs/learn/policies/microduck.md:91 names:
#
#     sim.run_policy(robot_name="microduck", policy_provider="microduck",
#                    policy_kwargs={"target_velocity": [0.15, 0.0, 0.0]})
#
# This is the sim-equivalent of the real-mode `send_action({"vx":0.1,
# "vyaw":0.0})` on the SAME robot and SAME ONNX file -- byte-for-byte, per
# docs/learn/hardware/microduck.md:46-56 ("difference 0.0"). The refusal
# above never names it.

print(">>> REPRODUCED <<<")
print("send_action({'vx':..,'vyaw':..}) on sim intent-level robots refuses")
print("with no hint that the same keys ARE the real-mode intent and no")
print("pointer to the sim-equivalent run_policy(target_velocity=...) path.")
sys.exit(0)
