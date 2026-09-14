### Fixed: `extra["sample_weighting"]` accepts the fields lerobot's config declares

The friendly `sample_weighting` dict is forwarded as
`SampleWeightingConfig(**dict)`, so its accepted keys are that dataclass's own
fields. They were written down beside it instead, in three places, and had
drifted from it: `extra_params` -- which the config's docstring names as where
"additional type-specific parameters" go, and therefore the only place a
weighting scheme's own knobs can be spelled -- was missing from all three.

The two consumers also disagreed about a key the list did not name.
`build_config` raised, listing five accepted keys of six. `build_command` filtered
the argv against a separate literal tuple and dropped anything else without a
word, so a run asked for `extra_params` -- or one with a typo such as `kapa` --
trained with the field's default and reported success. `validate()` graded neither,
so the preflight passed both.

The accepted set is now read off the installed dataclass, the way
`extra["reward_model"]` already reads each reward type's own config fields, and
`validate()` grades the keys once for both consumers. Non-scalar values render
through the same `_render_extra_value` every other `extra` flag uses, so a
dict-valued field survives the argv: the flags `build_command` emits, parsed with
lerobot's own decoder, now produce a config equal to the one `build_config`
builds, for all six fields. A field lerobot adds is configurable the day it lands,
and a key no field matches is refused before launch by name.
