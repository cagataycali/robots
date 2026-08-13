# artifacts: unresolvable / mistyped policy_provider

`capture.py` runs the same script in a worktree at `upstream/main` and in the
branch; each dump records its own tree so `compose.py` can assert the two halves
came from different code. `compose.py` re-derives every drawn number from the two
dumps and asserts the border is clean before saving.

- `policy_provider_envelope.png` - the figure
- `main.json` / `branch.json` - the measurements
- `mutate.py` / `mut.json` - the mutation table (5 of 6 caught here, 0 of 6 by
  the 334 pre-existing cases over the same modules)
