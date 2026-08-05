### Fixed: `StreamingDatasetReader.open` refuses a numeric knob it cannot honor

`tolerance_s`, `buffer_size`, `max_num_shards` and `seed` were forwarded raw
into `StreamingLeRobotDataset`, whose constructor validates only `repo_type`.
Every consumer of the four sits downstream of the call that returned, so an
unusable value surfaced late or not at all: measured against a locally cached
dataset, `max_num_shards=0` (and any negative) opened successfully and then
streamed **zero frames** with no error on any path, `buffer_size=0` raised
`high <= 0` out of NumPy part-way through iteration, `seed=-1` was refused by
NumPy in a message naming neither the surface nor the parameter, and
`tolerance_s=inf` made the delta-grid check `open()` runs for parity with the
materialized dataset accept an off-grid `delta_timestamps` - while
`tolerance_s=nan` or a negative one refused a perfectly on-grid one in a
message blaming the caller's deltas.

Each knob is now checked against a shared domain before the lerobot import,
so a caller mistake is reported the same way with or without the extra
installed: the two counts on `positive_count_error` (both are read as a
`range()` bound or an exclusive upper bound), the seed on
`non_negative_count_error` (the domain `TrainSpec.seed` already uses, where
`0` is simply a seed), and `tolerance_s` on the signed `finite_number_error`
plus a floor of its own, because `0` there asks for an exact grid match
rather than being degenerate. `dataloader(batch_size=...)` is unchanged:
torch refuses an unusable batch size at construction.
