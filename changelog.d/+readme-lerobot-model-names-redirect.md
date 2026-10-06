`create_policy()` now redirects the LeRobot model-type names README publishes
under one bullet (`act`, `pi0`, `smolvla`, `diffusion`, `molmoact2`, plus the
sibling architectures `pi05`, `pi0_fast`, `vqbet`, `tdmpc`) with the tailored
sentence the removed `groot` spelling already had: "<name> is a LeRobot model
type, not a factory provider; use `create_policy('lerobot_local',
policy_type='<name>', ...)`". Previously a reader of README.md:93 who tried
`create_policy("act")` hit the generic `Unknown policy provider` dump at
`policies/factory.py:453` whose difflib hint offered nothing.
