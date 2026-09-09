### Fixed

- **A rollout driven by the blocking `run_policy` now counts as in flight.** The
  MuJoCo engine derived that population from its background-future table alone,
  so a rollout started by `run_policy` - which drives on the caller's own thread
  and registers no future - was invisible to every reader of it while it drove
  the arm: `list_policies_running` answered "No policies running.", the mesh
  `status` command answered `idle`, the state topic published `active=false`, and
  the `{"action": "stop"}` fanout `Mesh.emergency_stop` broadcasts answered
  `{"ok": true, "stopped": [], "note": "no policies running"}` and halted
  nothing - scored as a peer that stopped, so the operator was told the fleet had
  halted while the rollout drove the arm for the remaining 8 of its 10 seconds.
  `stop_policy` was the one surface that also read the per-robot claim, so it and
  `list_policies_running` reported opposite facts about the same instant. The
  population now counts both launch shapes, spelled once, and every reader
  inherits it.
