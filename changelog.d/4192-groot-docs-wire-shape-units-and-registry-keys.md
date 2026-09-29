### Fixed: the groot page says which server flavour each request shape needs, and the registry names every keyword

The `groot` page's "Run it" example set no mapping, which sends the flat
`video.X` / `state.X` request that only a `--use-sim-policy-wrapper` server
accepts, under camera names (`front`) the official SO-ARM checkpoint does not
have, so it failed on both server flavours (`Observation must contain a
'video' key` / `Video key 'video.room' must be in observation`). The page now
states that a mapping selects the nested request the plain
`run_gr00t_server` accepts, that the flat request returns the model's own
keys, that `groot_version="n1.7"` is required for an N1.7 server, that
`reset(seed)` reseeds only through the tool's `deterministic=True` wrapper,
and that the chunk arrives in the checkpoint's own units (a degrees SO-ARM
checkpoint pins a radians MuJoCo `so101` under `status: success`). The
example maps `room`/`wrist` and the six `so101` actuators. `policies.json`
advertises `groot_version`, `observation_mapping`, `action_mapping`,
`language_key`, `timeout_ms` and `api_token` in `config_keys`, so the
dashboard's policy form can offer them, and names N1.7.
