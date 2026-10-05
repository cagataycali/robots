### Fixed: a pre-built policy_object beside policy_provider or policy_config is refused, not silently ignored

`run_policy`, `start_policy`, `eval_policy` and `evaluate_benchmark` used to run the
`policy_object` and drop a `policy_provider` / `policy_config` passed with it without
a word, so a provider name the provider-only path rejects still ran to
`status="success"`. The combination is now refused before anything is driven, with
a message naming what would have been ignored. A bare `policy_object=` (or one beside
the default `policy_provider="mock"`) runs as before. The `run_policy` agent tool,
which builds the policy once before a recording, now forwards only that object.
