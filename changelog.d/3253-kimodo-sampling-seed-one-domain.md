### Fixed: the Kimodo sampling seed gets one domain, at every surface that sets it

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

`get_actions` reached the same two outcomes without going near the config at
all. Its documented per-call `seed=` override is read straight from `kwargs`, and
`PolicyRunner.run` / `evaluate` forward `policy_kwargs` verbatim to every call,
so `get_actions(obs, prompt, seed=2.5)` then `seed=2.9` ran the sampler once
(seeds seen `[2.5]`, both keyed as `2`) and `seed=float("nan")` raised
`ValueError: cannot convert float NaN to integer` mid-rollout, from a message
that never mentions a seed.

`sampling_seed_error` owns the domain now, and all three surfaces that set a
seed consult it: `KimodoConfig` (however the value arrives - constructor,
`from_dict`, `from_json`, or a `KimodoPolicy(seed=...)` override),
`KimodoPolicy.reset`, which stores a per-episode reseed with
`object.__setattr__` and so never re-enters `__post_init__`, and
`KimodoPolicy.get_actions`, which reads the per-call override from `kwargs`.
Both `reset` and `get_actions` check before touching any state - `get_actions`
before the buffered-motion key is built - so a refused seed leaves the held
motion and the cursor as they were.

`diffusion_steps` and `guidance_scale` are the other two per-call overrides
`get_actions` documents, and they are left as they are: both are coerced on the
line that reads them, so the sampler and the key receive the same value and
neither can name a motion it did not produce. The seed was the one override
whose two uses disagreed.

Sign and magnitude stay out of the domain: `torch.manual_seed` honors a negative
seed and the key holds it unchanged, so a negative seed round-trips, and a seed
too wide is refused by the applier itself with a `ValueError` naming the
overflow. What is refused is the complementary set - seeds that fail where nobody
is looking, or that do not fail and name the wrong sample.
