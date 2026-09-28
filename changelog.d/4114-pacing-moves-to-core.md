### Changed: loop pacing moves to `strands_robots._pacing`, and a driver import no longer loads the mesh

`Ticker` and `sleep_penalty_s` are pure timing shared by drivers, the mesh,
teleop and the simulation rollout loops, so they now live in the core layer at
`strands_robots._pacing`. Importing the Crazyflie, G1, Go2 or Reachy DOA driver,
or the shared `PolicyRollout`, used to execute the mesh package `__init__` (12
mesh modules) just to reach the pacer; it now loads none, and
`tests/test_import_layers_are_a_dag.py` pins that no driver imports the mesh.

Deprecated: `strands_robots.mesh.pacing` re-exports both names with a
`DeprecationWarning` and is removed in 0.7. Import from `strands_robots._pacing`.
