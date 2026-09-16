### Fixed: gr00t_inference accepts a checkpoint directory in your own home

`hf_local_dir` under `/home` was refused on Linux as a protected host path -
including the tool's own default `~/.strands_robots/checkpoints` when spelled
out - and every `$TMPDIR` path was refused on macOS. A visible directory under
your own home, the tool's checkpoints dir, the Hugging Face cache and the
system temp dir are now admitted; a hidden home entry such as `~/.ssh` is
refused by name with the remedy, and `/etc`, `/root`, other users' homes and
the docker socket are refused exactly as before.
