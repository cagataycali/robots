### Fixed: the Kimodo sampling seed gets one domain, at both surfaces that set it

`KimodoConfig` gates every numeric knob it carries - `diffusion_steps`,
`guidance_scale`, `num_frames`, `transition_frames`, `native_fps`, `tracker_fps` -
and did not gate `seed`. That field is used twice, and the two uses disagreed:
the seed is handed to the sampler as itself, and it is also part of the key
`KimodoPolicy` identifies the buffered motion by, which coerces it with `int()`.

So a fractional seed named a different sample than the one it produced.
`seed=2.5` followed by `reset(seed=2.9)` both keyed as `2`, so the reseed read as
a cache hit: the sampler ran once for two requested seeds and episode two
replayed episode one byte for byte while reporting that a fresh seed had been
applied. `True` keyed as `1` and silently shared that motion. `nan` and `inf`
could not be keyed at all and raised out of the private key builder mid-rollout,
past the construction-time reporting the config module exists to give - and `inf`
is what a config file spelling `1e400` parses to.

`sampling_seed_error` owns the domain now, and both surfaces consult it:
`KimodoConfig` (however the value arrives - constructor, `from_dict`,
`from_json`, or a `KimodoPolicy(seed=...)` override) and `KimodoPolicy.reset`,
which stores a per-episode reseed with `object.__setattr__` and so never
re-enters `__post_init__`. `reset` checks before touching any state, so a refused
reseed leaves the held motion and the cursor as they were.

Sign and magnitude stay out of the domain: `torch.manual_seed` honors a negative
seed and the key holds it unchanged, so a negative seed round-trips, and a seed
too wide is refused by the applier itself with a `ValueError` naming the
overflow. What is refused is the complementary set - seeds that fail where nobody
is looking, or that do not fail and name the wrong sample.
