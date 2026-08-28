### Fixed: `strands-robots doctor` no longer prints EGL tracebacks beside "All checks passed"

`strands_robots.doctor.check_sim_smoke` drove `Robot('so100')` through `step` and
`get_observation` and never released it. That observation renders the robot's
cameras, so it opens a MuJoCo GL context; left to the finalizer, the context was
freed during interpreter teardown - after EGL had already been de-initialised -
and MuJoCo's own `Renderer.__del__` wrote an `Exception ignored in` traceback to
stderr.

The command still exited 0 and still printed `All checks passed`, so a healthy
install produced a verdict saying the setup was sound beside 1894 bytes of
`EGLError(err = EGL_NOT_INITIALIZED)`. That is the first command a new reader
runs to find out whether their machine is set up, and it is the one place the
answer is supposed to be unambiguous. Measured 6 of 6 runs on mujoco 3.5.0 (the
declared floor) and on 3.12.0, and 0 of 6 after the fix.

Nothing about the verdict changes: the check still reports the observation key
count, and the renderer was the only thing left behind. The scope was decided by
where the context is opened rather than by who notices it last - `Robot()` as a
module-level name happens not to reproduce it, because a function local becomes
garbage when the function returns and a module global does not, so the shape of
the caller was deciding whether a traceback appeared.

The sim is released in a `finally`, so the paths that opened the context and then
refused - an empty observation, an observation that raised - free it too. The
release verb is `cleanup` rather than `with`: `Robot()` resolves to a simulation
or to the hardware wrapper, and only the simulation implements the
context-manager protocol, so `with` would release one of the two things the
factory returns. A release that fails is reported rather than swallowed, because
a sim that cannot be released is a real defect on that machine and the verdict
this check reports covers the whole lifecycle it drives.

The two `Robot` doubles in the doctor suite gain the release verb the real one
always has. Without it the release would fail with `AttributeError`, and the
cells asserting `FAIL` for an empty observation would still have been green while
grading nothing.
