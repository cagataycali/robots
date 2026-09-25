### Docs: the WBC page keeps the controller, rollouts get their own page

`docs/policies/wbc.md` had grown to 2,351 words carrying two subjects: the
controller - install, the weights recipe, the constructor parameters and the
`WBCConfig` value domain, the locomotion goal kwargs, the 86-dim observation
layout and the actuator table - and how a rollout of it is run.

Split at that H2 boundary. `policies/wbc.md` (1,352 words) is the controller;
the new `docs/policies/wbc-rollouts.md` (1,122 words) is the running of it: the
torque shim `run_policy` installs on a position-servo scene and its
`wbc_install_torque_control=False` opt-out, the pelvis-mounted camera a walking
robot is recorded from, the upstream torque-deploy loop with its clips, and
layering a manipulation policy on the arms with `CompositePolicy`. Both pages
are inside the 1,500-word budget, the new page has a nav row, and the deep
links from `policies/overview.md` and `training/vla_workflow.md` follow the
sections they name.
