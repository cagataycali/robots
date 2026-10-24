### Fixed: a headless Linux host renders whatever is imported first

`import strands_robots` now picks `MUJOCO_GL` (EGL, else OSMesa) on a Linux host with no display, before anything
imports `mujoco`. Before, it was picked only when the first world was built, so a script, notebook or library that
imported `mujoco` earlier locked GLFW and `render()` failed with `gladLoadGL error` while the message claimed
`MUJOCO_GL=egl`. If `mujoco` is imported before the package, the error now says which backend MuJoCo really chose and
what to set. An explicit `MUJOCO_GL` is never overridden.
