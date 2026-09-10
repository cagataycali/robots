### Fixed

- The fleet examples' `[y/N]` HITL gates treat a closed stdin (CI, a pipe that ran dry, a detached run) as a decline that is printed and recorded, instead of dying in an `EOFError` traceback partway through a run.
- Examples default `MUJOCO_GL` to a backend the host actually has (`cgl` on macOS, `egl` elsewhere; an exported value still wins) instead of the Linux-only `egl`, which is `RuntimeError: invalid value for environment variable MUJOCO_GL: egl` at `import mujoco` on a Mac. The MUJOCO_GL linter now fails any example defaulting to `egl`/`osmesa` unguarded, in any scope.
- `examples/fleet/05_work_order_dispatch.py` writes its default `work_order_events.jsonl` under the temp dir instead of the current directory (`--events` still picks any path).
