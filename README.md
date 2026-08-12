# mesh scouting-default prose

`capture.py` measures `scouting_block()`'s emitted default and the four prose
sites on both trees (main via `git show`, the branch from the worktree), and
runs the new guard with the prose reverted to the merge base and restored.
`compose.py` renders the figure and asserts every rendered number against
`facts.json` before saving.
