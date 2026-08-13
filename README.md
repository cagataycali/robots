# One malformed `joint_command` must not discard the commands behind it

`capture.py` drives the cyclonedds `_poll_loop` with a **real MuJoCo so101**
behind it (the bridge's `_on_command` calls `robot.send_action`, so a shim that
forwards to a live sim makes the loop really move the arm). One batch is fed:
a valid pose, a malformed position, then the pose the operator meant to end at.

Run it once per tree (`PYTHONPATH=<tree> MUJOCO_GL=egl python3 capture.py <outdir>`),
then `compose.py` builds the figure and asserts every number it draws.
`mutate.py` is the 7-regression x 2-arm mutation table.

Measured: main applies 1 of the 2 valid commands and strands the arm at the
intermediate pose (worst joint 2.2359 rad = 128 deg from the commanded pose);
this PR applies both and reaches the commanded pose joint for joint.
