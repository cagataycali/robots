# add_camera pose-rule measurement

`deadcheck_probe.py` - neuters the second application of the pose rule and compares every
outcome (0 of 40 changed).
`capture.py` - per-tree capture: verdicts for 15 values x 2 pose parameters, the
neutering measurement, and one MuJoCo headless render through the surviving path.
`compose.py` - builds `artifact.png` and asserts every rendered number.
`mutation_table.py` / `mutation_table_faithful.py` - the mutation tables, including the
faithful re-insertion of the removed loop.
