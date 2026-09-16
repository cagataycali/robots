### Fixed: a rollout under `MUJOCO_GL=egl` no longer ends in an `EGLError` traceback

`mujoco.Renderer` frees its GL context in `__del__`, and under EGL the display
those contexts belong to is terminated by an `atexit` hook mujoco registers
when it first opens it. A renderer still alive at interpreter exit - the
main-thread cache of any script that never calls `cleanup()`, which is every
`Robot(...).run_policy(video=...)` script and every `get_observation()` on a
robot with cameras - was finalised after that hook, so `free()` raised
`EGLError: EGL_NOT_INITIALIZED` from `eglMakeCurrent`, once from the renderer
and once from its context: some thirty lines of ignored traceback after a
successful rollout (mujoco 3.13.0, seen on an L40S and a Jetson Thor).

The MuJoCo rendering mixin now records every renderer it builds with its
owning thread and registers one `atexit` hook after the first build - after
mujoco's, so it runs first - that closes the renderers owned by the exiting
thread while the display is still initialised. No cross-thread close is
attempted (a GL context is bound to its creating thread), and none is needed:
a worker's thread-local cache is dropped when the worker ends, before the
hooks run.
