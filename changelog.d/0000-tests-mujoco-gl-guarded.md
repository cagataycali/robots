### Fixed
- Twenty test modules that hard-defaulted `MUJOCO_GL=egl` at import now pick `cgl` on macOS like the rest of the tree, so any of them runs in isolation on a Mac instead of dying at `import mujoco` with `RuntimeError: invalid value for environment variable MUJOCO_GL: egl`.
