### Fixed: `create_policy("persistent")` says how to build it instead of a bare CPython `TypeError`

`persistent` resolves by module name, but `PersistentPolicy` needs its own
`provider` argument, which `create_policy` already binds. Every call failed as
`missing 1 required positional argument: 'provider'`, or `got multiple values
for argument 'provider'` when the caller passed one, naming neither the
provider nor the fix. `create_policy` now refuses it before construction with a
`TypeError` that names the provider and the direct construction
(`PersistentPolicy(provider="mock")`), and `policy_provider_error` reports the
same sentence to the agent tools. `provider` is now positional-only on
`create_policy`, so `provider=` reaches the refusal. A constructor argument a
provider requires and the call omits (`create_policy("composite")` without
`lower`/`upper`) is now named with the provider and what it accepts.
