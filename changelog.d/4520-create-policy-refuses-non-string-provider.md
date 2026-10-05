### Fixed: `create_policy(<non-string>)` is a `TypeError` naming `provider`, not a raw `.strip()` error

`create_policy(None)`, `create_policy(42)`, `create_policy(b"mock")` or
`create_policy({...})` failed inside resolution as `AttributeError: 'NoneType'
object has no attribute 'strip'` or `TypeError: unhashable type: 'dict'`,
naming neither the parameter nor the fix, while `provider_can_be_created`
already answered `False` for the same values. Resolution now refuses a
non-string first with `provider must be a string, got <type> (<value>)`, and
a `Policy` instance or class handed over by mistake is pointed at
`policy_object=`. `policy_provider_error` reports the same sentence.
