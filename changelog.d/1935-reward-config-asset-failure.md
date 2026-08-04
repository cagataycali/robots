### Fixed: a reward config that cannot obtain its pretrained asset reports the type and a remedy, and the parity suite no longer downloads one

`LerobotTrainer.build_config` translated one class of reward-config constructor
failure into an actionable error - a `TypeError` from a forwarded field becomes a
`ValueError` naming the rejected fields - and let every other class leak. A reward
config may derive a field from a pretrained asset inside its own `__post_init__`:
`robometer` reads its backbone's config and tokenizer to size `vlm_config`, so
merely *constructing* it downloads about 11 MB from the Hub. On a host that cannot
reach the Hub - offline, rate-limited, or behind a proxy - `build_config` therefore
failed with a bare `OSError: We couldn't connect to 'https://huggingface.co'`,
naming neither the trainer, nor the reward type, nor a way forward, and after
`validate()` had already returned no problems for the same spec.

That failure is now translated like its sibling: the error names the reward type,
quotes the underlying reason, states that the spec is not what is wrong (because
`validate()` has no network and cannot see this), and gives both remedies - make
the asset available, or pass the derived field through `extra['reward_model']` so
the constructor fetches nothing. `OSError` is the narrowest superset that covers
it: every `huggingface_hub` failure class for an unobtainable file is an `OSError`
subclass and `transformers` re-raises a plain one, while the `ValueError`s a config
raises for a bad field value are already actionable and are deliberately left
alone.

The parity tests hand `robometer` the field it would otherwise derive, so the suite
measures the discovery and passthrough contracts it is about rather than Hub
reachability. Both entry points into a backbone fetch are made fatal in a new case,
so a reward type that starts deriving a field from a pretrained asset must be given
that field rather than left to download it; the deriving path itself stays covered
by a case that skips when the backbone is not in the local cache.
