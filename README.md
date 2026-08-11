# Artifact: the ROS 2 bridge's missing stop tool

`capture.py` runs one agent session against `RosBridgedRobot` with the forwarded
`use_ros` call recorded, so it needs no ROS 2 environment. It is executed twice -
once against `upstream/main` and once against the PR branch - and each run writes
the tree it imported from into its JSON dump. `compose.py` renders the figure and
asserts every number in it against those two dumps before saving.

- `measured-main.json` - upstream/main
- `measured-pr.json`   - this PR
- `ros2_stop_tool.png` - the composed figure
