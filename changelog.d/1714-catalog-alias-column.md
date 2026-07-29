### Docs: the robot catalog lists every alias the registry accepts

The `Aliases` column in the `docs/robots/*.md` catalog tables showed at most three
aliases per row and dropped the rest with no "+N more" marker, hiding 19 names
`resolve_name()` accepts today -- including `franka_panda` and
`franka_emika_panda`, two of the first spellings a reader guesses for
`Robot("panda")`. Eight of the 72 rows truncated.

Each row now lists every alias its `robots.json` entry declares, in the order the
entry declares it, and `tests/test_docs_robot_catalog_coverage.py` compares the
column cell by cell so a newly registered alias cannot fall outside it. The two
rows that already listed a complete set in a different order were normalised to
the registry's order, so the column now follows one convention rather than two.
