### Fixed: no module-scope MUJOCO_GL default names a single platform's backend

MuJoCo validates `MUJOCO_GL` once, at `import mujoco`, against a table it builds
per platform, and `egl` is not in it on macOS: `RuntimeError: invalid value for
environment variable MUJOCO_GL: egl`. 27 module-scope defaults hard-coded `egl`
(18 under `tests/`, 2 under `tests_integ/`, 7 in `examples/`), so on a Mac each
of those modules died at import. They now use the guarded form the rest of the
tree already used, `"cgl" if sys.platform == "darwin" else "egl"`, which is
unchanged on Linux.

`tests/test_examples_mujoco_gl.py` owned this rule but graded only the *windowed*
backends, and its docstring recorded `egl` as a deferred question. It now grades
every backend MuJoCo names -- none of the four works on every platform -- so an
unguarded `egl` is reported exactly like an unguarded `cgl`, and a 21st such
module cannot be added. `tests/conftest.py` also picks a platform-valid default
before any test module is imported, so collection order can no longer decide the
session's backend.
