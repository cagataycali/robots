### Changed: `sync_dataset_to_bucket` lives in `strands_robots.dataset_transfer`

`strands_robots.sync_dataset_to_bucket` and `DatasetRecorder.sync_to_bucket`
keep their signatures and their behaviour; only the private module the function
is defined in changes, from `dataset_recorder` to the new `dataset_transfer`.

A bucket sync needs a finalized dataset directory and the `hf` CLI - no live
recorder, no sim world, no lerobot import - so it sits in `core` beside the
three dataset reads rather than with the recording session in `app`. One
deferred inversion goes with it: the backend-agnostic sim recording mixin no
longer defers an import of the recorder module to upload a directory the
recorder never saw.
