# Microduck build lane — LOG (loop l20260827212223001, macbook cagatay-80a997247062)

One line per iteration. Newest at the bottom.

- iter1 (LAYER A LANDED): #373 MicroduckPolicy provider built + PROVEN. Package
  strands_robots/policies/microduck/{__init__,_session,observation,policy,composite}.py.
  Byte-compat vs raw onnxruntime on alpha_walking.onnx = 0.0 max abs delta (61-D obs).
  Real MuJoCo rollout via Robot("microduck").run_policy(policy_object=..., 50Hz) moves
  joints, no crash. 21 microduck tests + all policy meta-tests green; ruff+format+mypy clean.
  Registered provider "microduck" (shorthands microduck/microduck_walk/microduck_stand);
  MicroduckPolicyBundle kept as a programmatic wrapper (NOT a registry provider, mirroring
  CompositePolicy) so it needn't shoehorn into the one-class registry model. Added
  [microduck] extra, docs/policies/microduck.md + mkdocs nav, examples/microduck/microduck_walk_sim.py,
  changelog fragment 373. Pinned meta-tests updated: MicroduckPolicy -> _MUST_VALIDATE +
  _OWNING_SURFACES; MicroduckPolicyBundle -> _MUST_FORWARD; _PROVIDER_POLICIES.
  Next: Layer B (#371 native driver over Pollen NDJSON-unix-socket).
