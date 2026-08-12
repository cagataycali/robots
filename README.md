# device_connect bring-up outcomes

`capture.py` drives `init_device_connect_sync` for all three bring-up outcomes and
reads back both what the direct caller receives and what the shipped foreground
runner (`strands_robots.robot._run_device_connect_foreground`) tells an operator.
It is run once per tree; each run records the tree it imported from.

* `main.json`   - upstream/main
* `branch.json` - this change
* `compose.py`  - builds the figure, asserting every rendered value against the
  two dumps, that the two runs resolved different trees, and that the panel has
  no content in its 8px border.
* `mut.py`      - the mutation table (5 regressions x 2 test arms), AST-scoped to
  `init_device_connect_sync` and restoring the source byte-identically.
