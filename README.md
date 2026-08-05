# val_episodes count domain - measured verdict table

`measure_val_episodes.py` is run once in a worktree at `upstream/main` and once on
the branch (each run records the tree it imported, and the figure generator asserts
the two differ). `compose_figure.py` renders the two dumps and asserts every number
it prints, including that the usable-count and unset rows are byte-identical across
the two trees.

Dataset: 10 episodes, 1 task. Contract divergences: 22 of 30 cells -> 0 of 30.
