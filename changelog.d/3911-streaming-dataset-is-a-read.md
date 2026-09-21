### Changed: `streaming_dataset` is graded as a `core` module, not an `app` one

No import path moves and no behaviour changes: `strands_robots.stream_dataset`
and `StreamingDatasetReader` keep their home and their signatures.

What moves is the layer the import grader (`scripts/check_import_layers.py`)
holds the module to. Streaming a dataset back reads one and writes none, so it
joins `dataset_metadata` and `dataset_source` in `core`, under the sim facade
that is its only consumer inside the package. One deferred inversion goes with
it: `simulation.recording` no longer reaches up a layer for the read-back of a
dataset it had just recorded.
