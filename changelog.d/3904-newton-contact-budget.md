### Fixed: a rendered rollout on the Newton backend keeps the contact budget its physics needs

Newton sizes the MuJoCo-Warp contact buffers by estimating from the state the
world was built in, so a rollout that reaches poses the initial one did not
overflows them: MuJoCo-Warp printed `broadphase overflow - please increase
nconmax to 63` every step, wrote past the buffer, and the next solver step
raised a raw `Warp CUDA error 700: an illegal memory access`. An `so101` + cube
scene with a camera aborted on the first `send_action` after a `render`, which
made every image-driven policy rollout on this backend unrunnable. The solver is
now built with a pose-independent floor (512 contacts / constraints, scaled up
with the scene's shape count), passed to any solver whose constructor accepts
the keywords.
