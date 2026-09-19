### Fixed: the Cosmos 3 sim-loop example installs and reports what it documents

`examples/vla/cosmos3_diffusers_mujoco_rollout.py` described a different script
from the one in the file, in three ways, each measured in a fresh environment
built from its own install line against real `nvidia/Cosmos3-Nano` weights.

Its line installed `[cosmos3-diffusers,cosmos3-sim]`, which declare neither
`robot_descriptions` (the Panda MJCF it loads) nor `imageio` (the encoder
`--render` writes through), and both imports sat *after* the model forward pass -
so the documented command spent the pipeline load plus sampling and then raised
`ModuleNotFoundError: No module named 'robot_descriptions'`. The line now names
`sim-mujoco`, which declares both, and every import the run needs happens before
the GPU work, so a missing distribution costs a second rather than minutes.

`os._exit(0)` after `--render` discarded block-buffered stdout: the Cosmos chunk
shape and the Cartesian tracking error - the two numbers the script exists to
report - never reached a redirected log, while the video was written and the
process exited 0. The exit flushes first.

`--steps` was parsed and never read. The sampler count is a
`Cosmos3DiffusersBackend` parameter and `Cosmos3Policy` forwards only
`embodiment`/`model`/`mode`, so every run sampled the backend default of 35
whatever the flag said; the example now builds the backend, and the flag's
default is that same 35.

Two graders keep the class out of the tree: every `os._exit` under `examples/`
and `strands_robots/` flushes stdout before exiting, and an example's documented
install line installs every distribution this project declares and the file
imports, with every advertised `--flag` read somewhere. Each failed on this file
before the fix and on nothing else.
