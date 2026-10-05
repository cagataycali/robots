### Fixed: `register_policy()` refuses a built-in provider name or alias

A runtime provider registered under a name or alias that `policies.json`
already uses (`register_policy("wbc", ...)`, or `aliases=["random"]`) silently
replaced the built-in for the rest of the process, so `create_policy("wbc")`
returned the stand-in with nothing logged. It is now a `ValueError` naming the
provider it would hide, and nothing is registered; `overwrite=True` shadows it
on purpose, as `register_robot()` already asks.
