### Added: a wrist camera, per-body randomization, `Perturb`, `sim.editing()`, a determinism flag

`sim.camera(..., attach="so101/gripper")` puts a camera on any link (`describe()` lists each robot's
`links`); a bare robot id still means its base. `randomize(target="cube", mass=..., friction=...)` scales one
robot or object instead of every body, and `randomize(camera=0.01)` moves every camera up to 1 cm per axis.
`Perturb(policy, joint_noise=, image_noise=, action_delay=)` gives a policy a real arm's noisy readings and late
actions. `with sim.editing():` batches scene edits into one rebuild, which on `physics="newton"` means one
kernel compile; a load slower than 5 s logs the compile time and `WARP_CACHE_PATH`.
`describe()["physics"]["deterministic"]` is true for MuJoCo and one-env Newton, false for a batch and Isaac.
A test pins that `Simulation("isaac")` never accepts the Isaac Sim EULA for the user; the isaac extra is
pinned to `isaacsim<6.2`, the version it was measured on.
