# pose_tool interpolation-option pacing

`main-measured.json` and `branch-measured.json` are the raw dumps behind the figure.
One measurement script was run twice - once inside a checkout of `upstream/main` and
once inside the branch, each run printing the tree it resolved `strands_robots` from -
driving `pose_tool(action="move_multiple", ...)` six times against a fake serial bus
and recording, per request: the returned status and text, every `(motor, degrees)`
goal position that reached the bus, and the delay each `time.sleep` was asked for.

The figure generator asserts every number it states before saving, including that
upstream/main's six rows are identical (42 writes, 1.110s, 0.05s delay) and that the
default-arguments row is byte-identical across the two trees.
