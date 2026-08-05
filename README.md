# Artifact: remote-inference port domains (strands-labs/robots#1952)

`measure1952.py` constructs `PolicyServer`, `RemotePolicy` and the `--port` CLI
with each candidate value and records the verdict. It was run in a
`git worktree` at `upstream/main` and in the branch tree; each run prints the
tree it imported so the two halves cannot be confused.

`fig1952.py` composes the two dumps. It asserts the measured counts
(`20 of 39` before, `0 of 39` after), that the two dumps came from different
trees, that the ephemeral bind returned a real OS-assigned port on both, and
that the rendered PNG has a clean border.
