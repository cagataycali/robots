### Fixed: the fleet dispatcher picks its execution seam by capability, not by backend name

`examples/fleet/01_skill_dispatch_multi_vendor.py` selected between its two
execution seams with `args.backend == "mujoco"`, so a backend that implements
the synchronized surfaces was still sent down the sequential per-robot
`run_policy` fallback: the `move_to`-bound staging skill executed as a policy
rollout instead of the IK primitive its binding names. Isaac has implemented
both `run_multi_policy` (#2158) and the motion primitives (#2155) since #2122
and #2123 closed. `choose_executor` now asks the backend what it implements -
the same rule the dispatch layer above applies to robots - so a backend that
gains parity needs no edit here, and Newton, which implements neither, keeps
the fallback. Measured on one MuJoCo scene, the staging TCP lands 13.3 mm from
its target through the primitive against 464.2 mm through the fallback. The
suite's prose (examples 01 and 04, `examples/fleet/README.md`) no longer
describes that parity as pending.
