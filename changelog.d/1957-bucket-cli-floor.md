### Fixed: the huggingface_hub floor bucket sync declares is the release that actually ships the `hf` bucket CLI

`sync_dataset_to_bucket` runs `hf buckets create` and `hf sync`. Both first ship
in **huggingface_hub 1.5.0** (`huggingface_hub/cli/buckets.py`, registered by
`cli/hf.py`); every earlier release installs an `hf` entry point without them and
answers either invocation with `Error: No such command 'buckets'`.

The version gate, the upgrade instructions it emits, and the `[wbc]` extra's pin
all named `>=1.0`, so a caller on 1.0-1.4.x passed the gate and received that raw
CLI usage noise - exactly what the gate exists to replace - and the remedy the
library printed (`pip install -U 'huggingface_hub>=1.0'`) could be followed to
the letter and still resolve a CLI that cannot run a bucket sync.

The floor is now a single constant the gate, both messages, the README guidance
and the `[wbc]` pin are all checked against, so they cannot drift apart again.
A fresh install is unaffected (it already resolved 1.26.0); the change is to what
a constrained or pre-existing environment is permitted to resolve.
