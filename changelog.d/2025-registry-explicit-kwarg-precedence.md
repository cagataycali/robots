### Fixed: `build_policy_kwargs` no longer discards a value the caller supplied

A provider's `defaults` in `policies.json` were merged into the kwargs dict
*before* the caller's `**extra`, and the `extra` loop skipped keys already
present -- so the default the previous loop had just inserted won, and the
value the caller passed under the provider's own key was dropped with no error
on any path. Every one of the twelve `(provider, key)` pairs where a provider
declares a default for a forwardable key was affected, including
`build_policy_kwargs("wbc", walk=False)`, which returned `walk=True` and so
handed back the locomotion controller to a caller who had asked for the
standing one.

The three merge sources are now applied in precedence order: a value the
caller supplied under the provider's own key in `extra`, then the generic
parameter that maps onto the same key (`policy_host` -> `host`), then the
registry default -- which now only ever fills a key the caller left unset.
This is the rule `resolve_policy` already applies with its trailing
`kwargs.update(extra_kwargs)`, and the rule the suite already asserted for the
generic parameters. The `config_keys` filter is unchanged: a key outside the
provider's `config_keys` is still dropped.
