### Fixed: `policies/vera` - the teacache threshold reaches the server in both launch modes

`VeraConfig` drives the VERA policy server through two vocabularies: the
environment overlay `server_env()` returns, and the server's own launch flags.
The subprocess runner composes those flags directly (`--sample-steps`,
`--teacache-thresh`, ...); the container path has to get each value in front of
the same flags indirectly, as `-e` variables `docker/entrypoint.sh` turns back
into argv. `teacache_thresh` was carried by neither half - the `docker run`
command named no variable for it and the entrypoint had no branch to emit it -
while its `teacache` off-switch was carried as `-e VERA_NO_TEACACHE=1`.

So one config ran the DiT cache at two thresholds depending on how the server
happened to be launched: `VeraConfig(teacache_thresh=0.25)` passed
`--teacache-thresh 0.25` under `server_mode="subprocess"` and started the
container on the server's own default under `server_mode="docker"`. Nothing
reported it. `docker run` has no opinion about an environment variable nobody
passed it, and the value that decides how often the DiT is recomputed is exactly
the quality/latency knob a caller reaches for after seeing a rollout - so the
mode that ignored it looked like the threshold having no effect.

Both halves now carry it: the container command forwards
`-e VERA_TEACACHE_THRESH=<value>` in the same either/or shape as the subprocess
argv, and the entrypoint translates it back into `--teacache-thresh`. An `-e`
nothing in the container reads would have been inert, so the regression test runs
the shipped entrypoint under bash with a stub interpreter and asserts on the argv
the server would really have been started with. The headline check derives what
must be carried from the subprocess argv itself, so a flag added to
`_build_command` later is graded on arrival; `--host` (the container binds
`0.0.0.0` by design) and `--algo-config` (a host path, which needs a bind-mount
decision) are the two stated exclusions.
