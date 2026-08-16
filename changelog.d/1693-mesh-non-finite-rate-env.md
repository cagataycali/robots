### Fixed: a mesh loop rate the environment cannot express falls back instead of removing the limit

Every mesh loop rate is operator-tunable through an environment variable, and
every consumer turns that rate into a period with `1.0 / hz`. `float()` accepts
`"inf"`, overflows `"1e999"` to `inf` and accepts `"nan"`, and neither survives
that division: `inf` gives a zero period, so a loop that meant to wait between
ticks never waits, and `nan` compares `False` against every bound, so a cap
built from it never trips. `Mesh._resolve_camera_hz` already refused non-finite
input for exactly this reason; the seven sensor publish loops and the teleop
apply-rate ceiling, which compute the same period from the same kind of value,
did not.

`STRANDS_MESH_POSE_HZ=inf` therefore left `_pose_loop` publishing to the mesh
as fast as the CPU allowed -- measured at roughly 400,000 samples per second
against the 10 Hz that variable documents -- and every other sensor topic
behaved the same way, while `nan` silently switched the topic off. More
seriously, `STRANDS_MESH_INPUT_MAX_HZ=inf` or `=nan` silently removed the
teleop apply-rate ceiling that exists so a peer streaming above the nominal
publish rate cannot slam servos into overcurrent, thermal or gear damage: a
200-frame burst was applied in full with `rate_dropped` left at 0. Only an
explicit `0` is documented to disable a rate limit.

The rule deciding which environment-held rates are usable now lives in one
place, `strands_robots.mesh.session.hz_from_env`, which reports a value no loop
can honor and leaves the fallback to each reader: a sensor loop keeps its
built-in rate, the camera loop stays off, the teleop ceiling reverts to its
default. The three readers can no longer diverge on what a rate is.
