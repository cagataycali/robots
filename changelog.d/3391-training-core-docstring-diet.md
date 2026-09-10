### Docs: the training core states the domain it checks, not the history of the check

`training/base.py` and `training/_validate.py` were 82% and 85% docstring: a
158-line module docstring that enumerated the shared validation gates as
ordinals ("the eleventh", "the twelfth" ...), 27 `Trainer._*_problems`
forwarders carrying 20-48 lines of narration each over a two-line lazy-import
body, and gate docstrings that restated the same `Args:`/`Returns:` block 27
times. A reader looking for a field's domain had to read a measurement of a
past run to find it.

The two files now say what each surface accepts. The five shared domains
(count, cadence, rate, weight, clip bound, unit interval) are stated once in
`_validate`'s module docstring and each gate names its field and its domain in
a line or two; the lazy-import rule and the reads-the-field scoping rule are
stated once on `Trainer` instead of once per method; `TrainSpec`'s `Attributes`
block keeps every field and its domain and drops the per-field history.
`RLTrainSpec.actor_obs_keys` and `critic_obs_keys` shared one combined entry
and now have one each. No code changed: with docstrings stripped, both files
are byte-identical to before.

`tests/training/test_spec_fields_are_documented.py` pins what the prose is for
- every `TrainSpec` / `RLTrainSpec` field has its own `Attributes` entry, and
every shared gate names the field it reports on - so a future distillation
cannot drop a field an agent has to populate or a knob a refusal names.
