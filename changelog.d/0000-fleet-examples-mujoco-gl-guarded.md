### Fixed
- The five fleet examples pick `MUJOCO_GL` the way the rest of the tree does (`cgl` on macOS, `egl` elsewhere, an exported value always wins) instead of hard-coding the Linux-only `egl` - on macOS every documented live command (`python examples/fleet/04_emergency_evacuation.py`) died at `import mujoco` with `RuntimeError: invalid value for environment variable MUJOCO_GL: egl`.
