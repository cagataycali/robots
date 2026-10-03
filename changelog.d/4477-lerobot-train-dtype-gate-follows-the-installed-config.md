### Fixed: the `lerobot_train` dtype pin follows the installed lerobot

lerobot main now declares `dtype` on `PreTrainedConfig` itself, so every policy,
ACT included, accepts `--policy.dtype`. The tool already gated the flag on the
installed config's fields and emits it correctly; the test that expected ACT to
be refused was pinned to lerobot 0.6.1 and failed against lerobot main. The
`dtype` and `gradient_checkpointing` pins are now one table-driven test whose
cases come from the installed registry, and the tool's docstring names ACT's
missing field as a lerobot 0.6.1 fact.
