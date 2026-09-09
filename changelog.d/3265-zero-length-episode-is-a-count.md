### Fixed: `verify-dataset` grades a run whose every episode recorded zero frames

`read_dataset_episode_indices` reports an empty `frames_per_episode` to mean the
episodes parquet carries no usable `length`, and scored that availability as
`any(f > 0 ...)`. So a collection run that wrote episode metadata and no frames
at all answered "this dataset carries no lengths", and `verify_dataset` reads an
empty list as nothing to compare and skipped its check 2 - the check whose whole
subject is a zero-length episode.

The gate was therefore non-monotonic in the damage it grades: a dataset holding
`[5, 0, 0]` frames named its two empty episodes and exited 1, while the strictly
worse `[0, 0, 0]` was certified `status="success"` (exit 0). The
`meta/info.json total_frames` claim beside it went unreported too, because check
3's frame comparison was gated on the parquet total being non-zero rather than on
a length being available - and a parquet whose every episode is empty sums to
exactly the zero that read as "no lengths".

Availability is now whether a length was *read*: a present column with a
recorded `0` is a frame count, so the empty episodes are named and the header
drift is reported. A column that is absent, or present but wholly null, stays
unavailable - a length nobody recorded is unknown, not zero - so neither gains a
zero-length verdict, and `min_frames=0` remains the documented way to skip the
check. `verify_dataset_episodes` keeps its count-only verdict and now shows the
recorded zeros in its `total_frames_per_ep` diagnostic instead of an empty list.
