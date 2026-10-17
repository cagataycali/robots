### Fixed: `Simulation()` without MuJoCo says how to install it

The first `Robot()` or `Simulation()` on an install without the `mujoco` extra used to fail with a bare
`ModuleNotFoundError: No module named 'mujoco'` from inside the backend. It now raises an `ImportError` that reads
`Simulation() needs MuJoCo: pip install 'strands-robots[mujoco]'`, the same hint every other extra gives.
