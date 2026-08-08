# Artifact: non-finite mesh env knobs

`capture.py` measures how every float-valued env resolver in
`strands_robots/mesh/` treats an unusable value, and what the resolved value
then does to the three safety comparisons it sizes. It was run unchanged in a
`git worktree` at `upstream/main` and on the branch; each run records the tree
it imported from, and `compose.py` asserts the two differ before drawing.

`compose.py` re-derives every rendered number from the two JSON dumps and
asserts the punchline (5 non-finite cells on main, 0 after) plus a clean
figure border.
