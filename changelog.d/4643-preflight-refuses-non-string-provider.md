### Fixed: the policy preflight helpers refuse a non-string provider instead of answering "no hook"

`preflight_policy`, `policy_overrides_preflight` and `preflight_reason` now raise
the same `TypeError` that `create_policy` raises for a provider that is not a
string (`provider must be a string, got NoneType (None) ...`). Before, they
swallowed it with every other resolution failure and answered `None` / `False`,
as if the provider simply had no preflight hook. Unknown provider names still
degrade to a no-op, as documented.
