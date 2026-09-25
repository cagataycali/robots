### Fixed

- `examples/02_policy_abstraction.py` listed the spellings `create_policy`
  accepts as a block of comments above a `Robot("so100")` sim, one of them
  `create_policy("allenai/MolmoAct2-SO100_101", embodiment="so_real")` - the
  hardware embodiment a sim example must not use. Measured on the sim `so100`,
  `so_real` declares `shoulder_pan.pos .. gripper.pos`, of which the sim
  observation binds 0 of 6, so `observation.state` is never composed; the
  robot's own `so100` binds 6 of 6. The same block named an ACT checkpoint that
  is not a repo and a policy class this package has no provider for. The rule
  that keeps a hardware embodiment out of a sim example now reads the recipes an
  example shows a reader as well as the calls it makes.
