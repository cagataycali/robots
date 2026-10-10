### Tests: thirteen more domain sweeps walk their value tables in one cell

The RL device, env numeric, seed and target-entropy training suites, the
lerobot_camera posture and vocabulary suites, the lerobot_train flag, size and
device suites, the pose_tool interpolation and smooth-flag suites, the WBC
dimension suite and the policy posture-flag suite gave every probe value its
own cell. Each test now walks its table in one cell and names the failing
value in the assertion. The thirteen files go from 1,092 cells to 357 with the
same package lines executed.
