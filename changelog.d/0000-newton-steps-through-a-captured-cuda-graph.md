### Changed: a Newton control step replays one captured CUDA graph, about 20x faster

Each Newton control step launched every kernel of its `substeps` solver steps
from Python. On an L40S with one so100, a step took 80 ms, so `run_policy` ran
at about 2 percent of real time: 30 steps took 49 s. On a CUDA device the step
is now captured once as a CUDA graph and replayed, which takes 3.3 ms per step
and 2.6 s for the same 30 steps. The trajectory is bit-for-bit the one the
Python loop produces, through `set_gravity`, `set_timestep`, `add_object` and
an odd substep count. The graph is captured again whenever a buffer it holds
changes. The graph is used with the `mujoco` and `featherstone` solvers, whose
replay was measured equal to the loop. `kamino` is left out: its replay drifted
from the loop by up to 2.4e-3 rad. Kamino, a CPU device, a failed capture, or
`STRANDS_NEWTON_CUDA_GRAPH=0` uses the Python loop. Joint targets are now
written into their existing array rather than a new one.
