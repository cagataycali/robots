### Fixed

- **cosmos3**: the de-normalization quantiles handed to `denormalize_quantile`
  (directly, or as `decode_cosmos_chunk_to_targets(stats=...)`) are held to the
  shared numeric-vector domain, in whatever spelling they arrive. Only the
  `action` was converted before, so the quantiles were dereferenced raw
  (`q01.shape[-1]`): a `list` — the layout the bundled `stats/*_stats.json`
  files store, and what a caller supplying an unbundled domain's own quantiles
  holds — raised `AttributeError: 'list' object has no attribute 'shape'`
  naming neither parameter, and pre-empted the width mismatch the comparison
  behind it exists to report. A `nan`/`inf` component was accepted and spread
  through every column of the chunk and every pose
  `decode_pose_trajectory` composed from it, returning an all-`nan` SE3
  trajectory as a successful result.
