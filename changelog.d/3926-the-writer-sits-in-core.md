### Changed: `DatasetRecorder` is a core contract, not an app host

`strands_robots.dataset_recorder` keeps its module path, its public names and its
behaviour; only the layer the import grader places it in changes, from `app` to
`core`.

A recorder is a writer a caller drives, not a host that composes the package: no
`app` module imports it, and no `app` module holds a recording session -
`start_recording` exists only on the three sim backends a layer below, and its own
five imports are all `core`. Placing it in `app` therefore inverted the layering
for its only in-package reader and made `recording_errors`, `dataset_source` and
`dataset_transfer` look like they had an `app` caller when that caller was the
recorder itself. It now sits in `core` beside the four dataset reads, and the last
`sim|policies -> app` inversion goes with it.
