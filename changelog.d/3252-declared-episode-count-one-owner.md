### Fixed: the episode count `meta/info.json` declares gets one owner

Five surfaces read that header and each graded it its own way, so one file got
five verdicts. `int(2.5)` truncates to `2` -- exactly the count a two-episode
parquet holds -- so a header no writer could have produced read as AGREEMENT
between the two independent metadata sources, and `verify_dataset_episodes`
returned `status="success"`, `sources_agree=True` on the inconsistent dataset
that cross-check exists to catch. `int(1e400)` raises `OverflowError`: `1e400`
is a well-formed JSON number that `json.load` parses to `inf`, so a perfectly
readable file raised out of three readers whose documented answer for an
unusable header is "unknown" -- and straight through a facade whose docstring
says a corrupt dataset is "reported as this same error dict, never raised".
`true` counted as one episode (`bool` is an `int` subclass) while the
`total_tasks` reader in the same module already excluded it, and `"2"` was two
episodes to three readers and unusable to the fourth.

`utils.declared_count` is that one owner: a declaration outside its domain
declares no count, never a nearby number. Because "no count declared" is also
what an ABSENT header means -- and an absent header is agreement, the parquet
being the sole truth then -- the readers that certify a dataset now report the
difference rather than collapse it: `read_dataset_episode_indices` carries it in
a new `info_problems` list, which `verify_dataset_episodes` surfaces as a
MISMATCH envelope, and `verify_dataset` appends a problem for `total_episodes`
and `total_frames` alike instead of silently skipping the comparison. A wrong
COUNT is still reported as drift with its existing message, and an absent header
still leaves the parquet as the sole truth.

Measured on one real two-episode MuJoCo recording across eleven header
spellings and all five readers: 25 of 55 verdicts were wrong before (four of
them a certified dataset, five a raised `OverflowError`), 0 after, with the
three healthy/unknown control rows unchanged.
