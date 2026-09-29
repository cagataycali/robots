### Fixed: `create_policy` reports a misspelled keyword before asking to trust remote code

On `kimodo` and `lerobot_local`, the two providers gated by
`STRANDS_TRUST_REMOTE_CODE`, a misspelled keyword such as
`create_policy("kimodo", model_i="...")` raised `UntrustedRemoteCodeError`
first. The caller was asked to opt in to remote code execution for a call that
would fail anyway, and saw the typo only on the second attempt. The keyword
check is pure, so it now runs before the gate and the typo is a `TypeError` on
the first call, as it already was for every ungated provider. A well-spelled
call still meets the gate unchanged.
