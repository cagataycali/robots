### Fixed: `get_observation(skip_images=True)` renders nothing while a recording is open

Every backend's `get_observation` turned `skip_images=True` back off whenever a
dataset recording kept cameras, so every caller that reads joints only paid for
the recording's cameras: a joint `stop_when` predicate, the undriven-robot
state, a twin or composite driver's joint read, and each non-first robot of an
Isaac `run_multi_policy`. A recorded MuJoCo rollout gated on a joint predicate
rendered every camera more than twice per step. The loops that record the
observation they read (`run_policy`'s observe step and the first robot of
`run_multi_policy`) now ask for the cameras themselves, so recorded frames keep
their image columns and `skip_images=True` means no render.
