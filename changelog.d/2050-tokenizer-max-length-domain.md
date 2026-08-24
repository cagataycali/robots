### Fixed: an instruction token budget the tokenizer cannot slice by is refused where it is named

`LerobotLocalPolicy` hands `tokenizer_max_length` to a HuggingFace tokenizer as
`max_length` alongside `truncation=True`, so the tokenizer reads it as a slice
bound over the encoded instruction. It was the one count in that constructor
stored verbatim: `actions_per_step` and `rtc_execution_horizon` are already
refused on arrival for exactly this reason, and `image_keys` and
`robot_state_keys` for the shape equivalent.

Measured against a real tokenizer on an 11-token instruction, `0` (and `False`)
produced a zero-width prompt - a language-conditioned VLA asked to act with its
whole task specification removed, with nothing reported on any path; `True` kept
only the first token; `None` fell back to the tokenizer's own 262144-token
ceiling for every inference step. The remaining values reached the tokenizer's
binding and surfaced as an `OverflowError` or `TypeError` naming neither the
parameter nor the policy, and only once inference had begun.

The count is now checked on the two surfaces its siblings already guard - the
constructor, before any checkpoint is fetched, and `preflight`, so the rollout
entry point reports it as a structured error - against the shared
`positive_count_error` domain, whose strict-`int` rule is what the tokenizer's
binding accepts.
