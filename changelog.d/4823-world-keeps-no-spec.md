### Fixed: a robot added to a MuJoCo world costs about its compiled model, not 330 MB

The MuJoCo backend no longer keeps the `MjSpec` after compiling the world; the spec held a second copy of every
mesh. Eight so101 arms in one world now take 1.2 GB resident instead of 3.0 GB (about 90 MB per added arm, from
about 330 MB). `scripts/profile_robot_rss.py` prints the curve.
