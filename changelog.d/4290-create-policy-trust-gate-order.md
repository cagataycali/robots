### Fixed: `create_policy` reports a misspelled kwarg before the trust-remote-code gate

`create_policy(provider, **kwargs)` ran the `STRANDS_TRUST_REMOTE_CODE` gate
before `policy_kwargs_error`, so for `kimodo` and `lerobot_local` a kwarg no
provider bound was masked by the security refusal. The caller was told to opt
in to remote code first, and only learned about the typo on a second call --
after having flipped a security-sensitive environment variable on a call that
would have failed regardless. Every other provider fires the kwargs check
first because it has no trust gate, so the asymmetry was invisible unless you
probed the two gated ones.

The docstring at `strands_robots/policies/factory.py:660` promised the
`TypeError` arrives "before construction, so no model is downloaded and no
server dialled on a typo"; that promise only held for the non-gated providers.
Now the kwargs check runs first (it is pure -- no import beyond what
`_resolve_policy_class` already did, no network, no code execution), and the
trust gate follows on a well-typed kwarg-bag that would otherwise proceed to
actual model loading.
