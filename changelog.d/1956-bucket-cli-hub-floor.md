### Fixed: bucket sync now requires the huggingface_hub release that actually has `hf buckets` / `hf sync`

`sync_dataset_to_bucket` shells out to `hf buckets` / `hf sync`, and those
subcommands first ship in huggingface_hub **1.5.0** - 1.4.1 carries no
`huggingface_hub/cli/buckets.py` and answers `hf sync` with
`Error: No such command 'sync'`. Every floor the project enforced or advertised
was `>=1.0`, five minor releases below that, so:

- the runtime version gate admitted 1.0.0 through 1.4.1 and the caller received
  the unroutable-subcommand noise verbatim - exactly the outcome the gate exists
  to replace with an upgrade instruction;
- the remedy it printed (`pip install -U 'huggingface_hub>=1.0'`) named a release
  that still cannot sync;
- the `[wbc]` extra - the project's only direct huggingface_hub pin, and one that
  does not pull lerobot - let a fresh resolve land on such a release.

The floor is now a single constant that the gate, both install remedies, the
`[wbc]` pin, the README guidance and the example all derive from, and the gate
compares all three version components so the floor release itself is admitted.
Resolution is unchanged (`>=1.0` and `>=1.5.0` resolve identically today); the
bump changes which releases are *permitted*, which is what a constrained or
pinned environment lands on.
