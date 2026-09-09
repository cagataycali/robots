### Fixed

- `list_policies_running` answers on every simulation backend. `docs/simulation/overview.md`
  lists it in the Policy action table with no backend qualifier, and documents `stop_policy` --
  a `SimEngine` contract -- as deriving its verdict from "the same in-flight population
  `list_policies_running` reads", so "the two never report opposite facts about the same robot
  at the same instant"; `docs/device-connect.md` makes the same promise for the Device Connect
  stop, whose sim driver runs on any backend. Only one of that documented pair existed off
  MuJoCo. Measured on a real Newton world driving `so101`, in all three phases of a rollout:
  `AttributeError: 'NewtonSimEngine' object has no attribute 'list_policies_running'`.

  The population the verb needs is already on the ABC -- `SimEngine._rollouts_in_flight`, the
  seam the mesh's own reporting surfaces read -- so the verb is answered there and every backend
  inherits it. MuJoCo's copy is removed rather than kept beside it: its override of that seam
  delegates to `_active_policy_robots`, so the population, and the Future prune that reader
  performs, are unchanged. MuJoCo's answers are byte-identical in all three phases, and the
  whole change is +2 executable statements (`base.py` +7, `mujoco/simulation.py` -5).

  A backend that reports no population at all is now refused rather than reported as idle.
  "No policies running" is an affirmative claim about every robot in the world, and a backend
  that cannot enumerate its rollouts has no evidence for it -- the same reason the state topic
  omits its `active` flag instead of publishing `false`. The refusal names the backend and the
  seam to override, mirroring `stop_policy`, which declines rather than reporting a halt it
  cannot stand behind. That branch is reachable today: an Isaac engine reports no population
  before `create_world`, because there is no world whose rollouts could be enumerated.
