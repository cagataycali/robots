### Fixed

- The fleet examples' `[y/N]` HITL gates treat a closed stdin (CI, a pipe that ran dry, a detached run) as a decline that is printed and recorded, instead of dying in an `EOFError` traceback partway through a run.
- Examples default `MUJOCO_GL` to a backend the host actually has (`cgl` on macOS, `egl` elsewhere; an exported value still wins) instead of the Linux-only `egl`, which is `RuntimeError: invalid value for environment variable MUJOCO_GL: egl` at `import mujoco` on a Mac. The MUJOCO_GL linter now fails any example defaulting to `egl`/`osmesa` unguarded, in any scope.
- `examples/fleet/05_work_order_dispatch.py` writes its default `work_order_events.jsonl` into a private per-run directory (`mkdtemp`, mode 0700) under the temp dir instead of the current directory. It is not a fixed name in the shared temp dir: `emit_event` appends through symlinks, so a predictable path is one another user of the host could pre-create. `--events` still picks any path.
