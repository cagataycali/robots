### Fixed: the CPU software-rasterizer warning names the cause that is still standing

`MUJOCO_GL=egl` falling back to Mesa `llvmpipe` was always reported as a missing
NVIDIA EGL vendor ICD, so a host that had already registered one - and a host
with no NVIDIA EGL library at all, where Mesa is the correct backend - were both
told to apply a fix that could not help. The warning now narrows the remedy with
the two host facts the backend already establishes for ICD staging: no NVIDIA EGL
library, an unregistered ICD, or a driver that is installed and registered but
unreachable from this process. `docs/troubleshooting.md` records the same three
causes.
