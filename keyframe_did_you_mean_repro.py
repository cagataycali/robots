"""Repro: Robot(..., keyframe=<typo>) refusal has no 'Did you mean' hint.

docs/learn/hardware/franka.md promotes `Robot("panda", keyframe="home")` as the
documented API for spawning a canonical pose. A one-character typo like
`keyframe="hom"` gets a bare "Keyframe 'hom' not found in 'scene.xml'.
Available: 'home'." - the sibling difflib pattern (used in 16+ sites of
strands_robots including spec_builder.py:194, base.py:_unknown_robot_msg,
and 10+ others) is absent.

Expected: "Did you mean 'home'?" hint when a close match exists.
Actual:   bare "Available: ..." dump.

Reproduce:
    $ MUJOCO_GL=egl python keyframe_did_you_mean_repro.py
    typo='hom':     Keyframe 'hom' not found in 'scene.xml'. Available: 'home'.
    typo='Home':    Keyframe 'Home' not found in 'scene.xml'. Available: 'home'.
    typo='start':   Keyframe 'start' not found in 'scene.xml'. Available: 'home'.

Upstream raise site:
    strands_robots/simulation/mujoco/simulation.py:3034
    strands_robots/simulation/mjlab/simulation.py:1211 (sibling backend, same gap)

Sibling difflib precedent (same codebase, same pattern):
    strands_robots/simulation/mujoco/spec_builder.py:194 (shape typos)
    strands_robots/simulation/base.py:_unknown_robot_msg (robot typos)
    strands_robots/policies/factory.py:401 (policy provider typos)
    strands_robots/registry/policies.py:555 (HF org typos)
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot

TYPOS = ["hom", "Home", "start", "default", "rest", "init"]

for typo in TYPOS:
    try:
        Robot("panda", keyframe=typo, mesh=False)
        print(f"typo={typo!r}: OK (unexpected)")
    except Exception as e:
        msg = str(e)
        has_hint = "did you mean" in msg.lower()
        print(f"typo={typo!r:>10}  did_you_mean={has_hint}  msg={msg}")

# ---- Sanity: happy path still works
r = Robot("panda", keyframe="home", mesh=False)
print(f"\nhappy path: Robot('panda', keyframe='home') -> {type(r).__name__} OK")
