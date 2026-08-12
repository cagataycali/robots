# Artifact: a floating base spawned at an undefined pose

`capture.py` builds the reported scene (two parked arms, then a Unitree Go2 added
with `keyframe="home"`), renders at t=0 and dumps every measured number to JSON.
It is run unchanged in a worktree at the base commit and in the branch;
`compose.py` builds the figure and asserts every value it prints against the two
dumps, including that the two runs resolved different trees.

`framing_sweep.py` is the camera choice: it renders both measured base poses from
one tree over five candidate cameras and reports the differing-pixel fraction, so
the framing is measured rather than guessed (the chosen camera scores 24.72%).
