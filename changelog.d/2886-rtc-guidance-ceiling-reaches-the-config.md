### Fixed: the RTC guidance ceiling reaches the config lerobot reads

`LerobotLocalPolicy` takes two Real-Time-Chunking overrides side by side, and
only one of them could take effect. `rtc_execution_horizon` is a keyword argument
of lerobot's `predict_action_chunk`, and `_predict_with_rtc` forwards it per call.
`rtc_max_guidance_weight` is not: lerobot's denoiser reads that ceiling off
`self.rtc_config.max_guidance_weight`, and the RTC keyword contract -
`ActionSelectKwargs` - carries `inference_delay`, `prev_chunk_left_over` and
`execution_horizon` and nothing else. So a ceiling kept on the policy object had
nowhere to go.

`_init_rtc` read the checkpoint's value as a *default* into
`self._rtc_max_guidance_weight` and never wrote a caller's override back. The
override's only remaining reader was one `logger.info("... max_guidance_weight=%.1f")`
line. Measured against lerobot's genuine `RTCConfig(max_guidance_weight=10.0)`,
asking for `2.0` left `rtc_config.max_guidance_weight` at `10.0` while the policy
attribute read `2.0`: a checkpoint tuned for a tight ceiling ran the model's own,
and the INFO line reported the number that was not in force. The override is now
written onto the config, which is the field the clamp
(`torch.minimum(guidance_weight, max_guidance_weight)`) and the
`nan_to_num(..., posinf=max_guidance_weight)` fallback both read.

Writing it there is what makes the second half necessary. The value had no domain
while `actions_per_step`, `tokenizer_max_length` and `rtc_execution_horizon` in
the same constructor all have one, so `0`, `0.0`, `-3.5`, `nan`, `inf`, `True`,
`False`, `"2.0"`, a list and a dict were all accepted. Two of those - `0` and a
negative - are values lerobot's own `RTCConfig.__post_init__` refuses outright
with "max_guidance_weight must be positive". That constructor never sees this one,
because the override lands on an already-built config, so the domain is asked
where the caller names the value instead: `positive_finite_number_error` at the
constructor, and at `preflight` beside the four sibling knobs it already checks,
so a rollout gets a structured error before the weight download rather than a
raise after it. An `inf` ceiling clamps nothing and a `nan` ceiling makes every
clamped weight `nan`, which is why finiteness is part of the domain rather than
just positivity.

`None` keeps its documented meaning - adopt the checkpoint's own value - and that
branch does not write to the config, so a policy that asked for nothing still
leaves the config it read from untouched. Moving the write ahead of that default
resolving trips four of the pre-existing RTC cells, which is the boundary this
change is careful not to cross.

Why the suite was green over it: `test_rtc_user_overrides_config_values` is named
for both knobs overriding the config and asserts only that
`policy._rtc_max_guidance_weight` keeps the value it was given - an attribute
round-trip, which held either way - while `TestRTCConfigSchemaContract` pins that
lerobot still *exposes* the field, grading the read and never the application. A
new cell asserts the ceiling is absent from `ActionSelectKwargs`, so if lerobot
ever accepts it as a keyword argument the author is told to forward it instead of
writing the config.
