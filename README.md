# Artifact: what reached the lerobot command line

`measure_argv.py` is run once inside a pristine `upstream/main` worktree and once
inside the branch worktree (each prints the tree it resolved, and the composer
asserts the two differ). `compose_figure.py` re-derives every number in the
figure from the two JSON dumps and refuses to save if any claim is stale.
