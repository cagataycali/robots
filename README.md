# examples/04_mesh_peer_discovery.py — mesh teardown measurement

`capture.py` runs once per tree (upstream/main worktree and the PR branch) and
records three things: the example run as a user runs it (exit code + wall
clock), what its cleanup line actually releases in-process (which attribute is
read, whether that read returns `None`, and how many non-daemon threads survive),
and a headless MuJoCo render of the world the example builds.

`compose.py` builds `mesh-teardown.png` and asserts every number it draws
against `facts-base.json` / `facts-pr.json`, including that the two arms
measured different trees and that the two renders agree to `max|delta| <= 2`.

Measured on Thor (aarch64, MUJOCO_GL=egl):

| property | upstream/main | this PR |
|---|---|---|
| script terminates | no (SIGKILL at 30.1s) | yes, exit 0 in 1.5s |
| attribute read | `"_mesh"` | `"mesh"` |
| read returns None | yes (cleanup skipped) | no |
| surviving non-daemon threads | 6 (`pyo3-closure`) | 0 |
| sim render | \| max delta \| = 1/255 vs the PR (unchanged) | |
