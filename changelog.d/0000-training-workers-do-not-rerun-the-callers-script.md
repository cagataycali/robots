### Fixed: in-process training no longer re-runs the calling script once per worker

`train_policy(provider="lerobot_local")` runs LeRobot's training in the calling
process, and LeRobot spawns 4 DataLoader workers by default. A spawned process
re-imports the parent's `__main__`, so an unguarded script (the shape of every
agent script) ran from the top again in each worker. A marker at the top of a
script ran 5 times for one training call. Every model call, simulation and,
with `mode="real"`, hardware command made before training was repeated once per
worker. Processes spawned by `call_callable` and `elastic_launch_callable` (the
LeRobot and Cosmos 3 in-process paths) now start without the caller's
`__main__`, and it is restored when the call returns. `spawn` is kept. Set
`STRANDS_TRAIN_WORKERS_IMPORT_MAIN=1` for Python's default behaviour.
