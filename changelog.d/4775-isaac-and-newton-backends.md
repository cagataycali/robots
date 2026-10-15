### Added: Isaac Sim and Newton backends

`Simulation("isaac")` builds the same world on Isaac Sim (PhysX): robots are imported from their MJCF into USD,
one `SimulationApp` and one open world per process, and the shared-world semantics hold (add, remove, reset and
reload keep time exact; a removal rebuilds the PhysX views). `Simulation("mujoco", physics="newton", num_envs=N)`
steps the MuJoCo world on the GPU with MuJoCo-Warp, N copies at once. `Robot.run` and `Simulation.run` drive both
unchanged. Neither backend imports a GPU stack until it is asked for, and each names its install when it is
missing. New extras: `[isaac]` (Python 3.12+) and `[newton]`; both need CUDA.
