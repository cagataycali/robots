### Fixed: the rollout tool's parquet-truth counts are graded, not coerced

`run_policy`'s parquet-truth gate exists to catch a fabricated episode count: it
reads `meta/info.json` after `stop_recording` and reports `total_episodes` /
`total_frames` as ground truth instead of trusting the loop's self-report. It
read both headers through `int()`, which is the one thing that gate cannot
afford, because a coerced number is what it then reports to the caller AS the
truth it independently verified.

`total_episodes: true` answered a one-episode request with `status=success` and
`parquet-truth: total_episodes=1`, no warnings -- a boolean certifying the count.
`2.5` truncated to `2` and `"2"` parsed to `2`, so each agreed with a two-episode
request and the mismatch guard had nothing to compare. Every one of those files
is already "metadata is corrupt" to `verify_dataset` and to
`read_dataset_episode_indices`, which read the same two headers: one file, three
answers. In the other direction `1e400` (a well-formed JSON number `json.load`
parses to `inf`), `NaN` and `null` raised `OverflowError` / `ValueError` /
`TypeError` out of a function documenting that it "returns a partial result on
missing fields rather than raising", and past the tool envelope a caller reads.

Both headers now go through `declared_count`, the one owner every reader of this
file already shares. A declaration outside that domain is "no count declared":
the count reads `-1` and the header is reported as corrupt metadata in the same
words `verify_dataset` uses, so it flips the tool's status to `error` instead of
certifying the rollout. The fabrication guard still fires on a genuine count that
disagrees, and no longer names the `save_episode` boundary for a corrupt header
-- that boundary may have fired for every episode. A header nothing declares is
reported as unverifiable rather than as a mismatch, and a `meta/info.json`
holding a JSON array instead of an object is an unreadable header rather than an
`AttributeError`.
