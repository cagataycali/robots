### Fixed: importing the package no longer pulls the simulation IK chain in through cosmos3

`strands_robots.policies.cosmos3` re-exported its MuJoCo bridge (`MinkIKBridge`,
`decode_cosmos_chunk_to_targets`) at import time, and that module imports
`strands_robots.simulation.ik`. The package root imports `policies` eagerly, so
`import strands_robots` loaded the IK chain in every process - including ones
that only wanted the Cosmos 3 network client. The two names now resolve on first
attribute access; `from strands_robots.policies.cosmos3 import MinkIKBridge` is
unchanged.

Without the `sim-mujoco` extra, `import strands_robots` now loads no
`strands_robots.simulation.*` module at all: 24 package modules instead of 49.
With mujoco installed the root still imports `simulation.mujoco.backend` on
purpose, to pin `MUJOCO_GL` before the first `import mujoco`, so there the saving
is the IK chain itself (51 -> 49 modules).
