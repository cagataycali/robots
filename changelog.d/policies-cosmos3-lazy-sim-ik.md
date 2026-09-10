### Fixed:

`import strands_robots` no longer loads the simulation package. `strands_robots.policies.cosmos3` re-exported its MuJoCo bridge (`MinkIKBridge`, `decode_cosmos_chunk_to_targets`) eagerly, and that module imports `strands_robots.simulation.ik`; the package root imports `policies` eagerly, so every process paid for 25 simulation modules it never asked for (median import 128 ms -> 103 ms on an M-series Mac, 50 -> 25 package modules). The two names now resolve on first attribute access; `from strands_robots.policies.cosmos3 import MinkIKBridge` is unchanged.
