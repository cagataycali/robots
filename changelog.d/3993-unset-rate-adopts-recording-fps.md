### Docs: an unset `control_frequency` adopts the open recording's fps

`start_recording()` then `run_policy()` with no rate once refused itself on the
two colliding library defaults (30 fps against 50 Hz). `_resolve_control_frequency`
removed that -- an unset rate now adopts the fps of a recording that is already
open, falling back to `DEFAULT_CONTROL_FREQUENCY` only when none is -- and
`docs/recording.md` plus `mujoco/tool_spec.json` state the rule. Five
caller-facing surfaces still warned about "the 50 Hz default rollout" being
refused with the episode landing at zero frames, and two named `50.0` as the
parameter's default where the signature declares `None`:
`examples/03_record_dataset.py`, `examples/07_post_tune_any_policy.py`,
`examples/notebooks/05_streaming_data_loop.ipynb`, `docs/simulation/newton.md`
and `docs/training/overview.md`, plus the params table in
`docs/simulation/rollouts.md`.

Each now states the adoption, the two examples and the notebook drop the
explicit rate they no longer need (the recorded dataset is unchanged: 100
frames, a 3.3 s span at a mean 0.03333333 s timebase, and the same frame bytes),
and the refusal of a rate the caller *does* pass is still documented, because
that one is still refused. `tests/test_examples_recording_rate_matches_rollout.py`
already graded the rates these examples state in code; it now also grades the
prose, so a surface naming a number for the unset rate fails rather than
drifting.
