### Fixed: a dataset metadata file that holds no JSON object is answered, not raised

Four readers of a dataset's `meta/*.json` promise to answer their documented
unknown rather than raise, and all of them graded that file by whether it could
be READ. A document that parses and is not an object - a bare `null`, an array,
a scalar, the shape a partially-synced or foreign file arrives in - carries no
headers at all, and `.get` on it raises `AttributeError`, a type none of those
handlers name. `verify_dataset` escaped as a traceback from the very corruption
it exists to report, discarding every problem already found and turning a CLI
whose contract is "exit 0 or 1" into an exit-2 stack trace;
`_dataset_codebase_version` aborted the whole `LerobotTrainer.validate()`
preflight; and `_dataset_quantile_stats_present` answered `False` - the DEFINITE
miss that refuses a QUANTILES-normalizing run - from a file it could not use.
Each now answers the same unknown its unreadable-file sibling does, and
`verify_dataset` reports what the file holds. Reading `meta/info.json` in the
checker has one owner, so the drift check reports the corruption and the video
check discards it rather than naming one broken header twice. The rollout tool's
parquet-truth gate and the judge's reader already graded the parsed document
this way; this is the same verdict at the readers that did not.
