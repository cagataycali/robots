### Removed: the `norm_stats.json` fallback in the lerobot_local processor bridge

`ProcessorBridge` carried its own MolmoAct2 normalizer: a port of lerobot's
`_FeatureNormalizer` that parsed a checkpoint's `norm_stats.json` and built
quantile/min-max/mean-std pipelines when no `policy_preprocessor.json` was
present. No checkpoint reached it.

A MolmoAct2 checkpoint is transformers-native, so `_load_model` routes it to
`build_policy` and returns before `_load_processor_bridge` ever runs; the
pipelines it uses come from lerobot's own factory
(`make_policy_config` -> `make_pre_post_processors`), which reads
`norm_stats.json` itself and resolves `norm_tag` against the tags the file
declares. The fallback's payload check accepted only that MolmoAct2 schema, so
it could fire solely for a checkpoint declaring a non-MolmoAct2 lerobot policy
type while shipping MolmoAct2-schema statistics -- a combination no published
checkpoint has, and one where MolmoAct2's transform would be the wrong one to
apply. Two implementations of one numeric contract, only one of them exercised,
is a silent-drift risk: lerobot's normalizer could change without anything here
failing.

The module, the fallback, its `norm_tag` parameter on
`ProcessorBridge.from_pretrained`, and the `inert_reason` reporting the fallback
was the sole producer of are gone. `norm_tag` remains on `LerobotLocalPolicy`,
where lerobot consumes it. The in-model normalization recovery for
pre-processor-era checkpoints -- which is reachable, and already delegates to
lerobot's `extract_normalization_stats` -- is unchanged, as is the `revision`
threading into every pipeline load.

`docs/policies/lerobot-local.md` described a priority order whose second entry
the MolmoAct2 family never took; it now documents the two bridge paths that
exist and states that MolmoAct2 normalization is lerobot's.
