### Fixed: `create_policy` refuses a `policy_config` bag passed as one keyword

`create_policy("mock", policy_config={"amplitude": 0.9})` handed the whole bag
to the provider's `**kwargs` sink, so the policy ran on its defaults - typo or
not - under `status="success"`. The rollout entry points unpack their
`policy_config`; a direct call now gets the same answer as a misspelled keyword,
naming the fix: `create_policy(provider, **policy_config)`.
