# artifacts/iot-provisioning-flag-boolean-domain

`capture.py` measures every `mesh.iot` posture flag on whatever tree it is run
in and dumps JSON; it was run once in a `git worktree` at `upstream/main`
(`measured_main.json`) and once on the branch (`measured_branch.json`).
`compose.py` builds `artifact.png` from those two dumps and asserts every
rendered number against them, including that the two runs resolved to
different trees.
