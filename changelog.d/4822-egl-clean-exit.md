### Fixed: a script that rendered exits without EGL tracebacks

With `MUJOCO_GL=egl`, every script that rendered ended with `EGL_NOT_INITIALIZED` tracebacks from
`Renderer.__del__`: MuJoCo registers `eglTerminate` at exit when it makes its first context, after the
renderers' release hook, so the display was gone before the contexts were freed. The release is now
re-registered whenever a renderer is made, so it runs first. Nothing is silenced; a GL error during a run still raises.
