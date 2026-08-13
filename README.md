# provider-import-remedy

Measured artifact for the policy-provider import-error fix.

* `capture.py` - run in a tree; reports the census, the recording report and the
  error a caller receives with `torch` reported absent exactly as the import
  system reports it. Run once at `upstream/main` and once on the branch.
* `compose.py` - builds the figure; asserts every rendered value against the two
  JSON dumps and that the two arms measured different trees.
* `mutation_table.py` - 6 regressions x 2 arms, AST-scoped anchors, byte-identical restore.
* `facts_main.json` / `facts_pr.json` - the two dumps the figure is built from.
