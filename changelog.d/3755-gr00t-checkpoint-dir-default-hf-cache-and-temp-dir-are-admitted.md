### Fixed: gr00t_inference accepts its own checkpoint directory and the system temp dir

`hf_local_dir` under `/home` was refused on Linux as a protected host path -
including the tool's own default `~/.strands_robots/checkpoints` when spelled
out - and every `$TMPDIR` path was refused on macOS, where the temp dir lives
under `/var`. The tool's checkpoints dir, the Hugging Face cache and the
system temp dir are now admitted. Nothing else under a home is: `hf_local_dir`
is agent-supplied and the mount is read-write, so `~/checkpoints` stays
refused, now naming the three places a checkpoint may go instead of "/home is
protected". `/etc`, `/root`, other users' homes and the docker socket are
refused exactly as before - including when the environment makes one of them
your home or your temp dir. Deciding a mount no longer creates the checkpoints
directory as a side effect, so an unwritable home reports a refusal instead of
raising.
