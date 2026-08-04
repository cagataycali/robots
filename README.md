# robot_mesh numeric-option domain — measured artifact

`artifact_measure.py` drives the real `robot_mesh` tool with a recording mesh
stand-in and dumps every verdict to JSON. It was run twice, once in a worktree at
`upstream/main` (`before.json`) and once on the branch (`after.json`); each dump
records the tree it resolved, and `artifact_figure.py` asserts the two differ
before composing anything. Every number in the figure is re-derived from those
two files and asserted in the generator.
