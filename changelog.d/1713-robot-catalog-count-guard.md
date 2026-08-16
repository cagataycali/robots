### Docs: the robot catalog now agrees with the registry, and a guard keeps it that way

`registry/robots.json` holds 72 robots in 8 categories, but nothing tied the
documented counts to it, so they drifted independently: the hero and
architecture SVGs said "40+ robots", the README said "50+", and
`docs/robots/index.md`, `docs/architecture.md` and the quickstart said 68. Four
category cards and the arms page understated their categories, and
`docs/architecture.md` quoted 106 aliases where the registry declares 114.

Two robots -- `hope_jr_hand` and `lekiwi_client` -- were missing from every
catalog table, so a reader had no way to discover names `Robot()` accepts. Both
are now listed with the existing hardware-only row convention.

Every count is corrected, and `tests/test_docs_robot_catalog_coverage.py`
derives all of them from `robots.json`: the catalog must list exactly the
registered names on the page for each robot's category, each stated count must
match the registry (the deliberately round "N+ robots" claims are pinned to the
current multiple of ten so adding one robot needs no copy edit), and a count
written anywhere the guard does not already know about is refused rather than
becoming the next stale number.
