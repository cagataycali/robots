# Sweeping the open set for untested compositions

Artifacts for strands-labs/robots PR "feat(ci): sweep the open pull request set for
untested compositions".

- `sweep-open-set.png` — the composed figure.
- `capture.py` — runs the sweep against the live API and records every number to
  `facts.json`. Asserts the mechanism it claims: that neither reported pull
  request's own single-branch run names either shared path.
- `compose.py` — draws the figure. Asserts every drawn number against
  `facts.json`, checks the row pitches, and checks the 8-pixel border is white.
- `mutate.py` — the mutation table: five regressions against the new cases and
  against the 30 pre-existing ones.
- `facts.json` — the raw capture.

Reproduce:

    PAT_TOKEN=... python3 capture.py facts.json
    python3 compose.py facts.json sweep-open-set.png

`capture.py` imports `scripts/check_merge_base_overlap.py` from the repository it
sits in, so run it from a checkout of the branch.
