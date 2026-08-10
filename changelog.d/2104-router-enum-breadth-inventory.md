### Docs: the actions the MuJoCo router dispatches but the schema never advertises are declared

`_dispatch_action` resolves an action with `getattr(self, name)` and no
allowlist, so every public method on `MuJoCoSimEngine` or its mixins is
dispatchable, while `tool_spec.json`'s `action` enum advertises a curated
subset. The enum-to-method direction was already pinned; the reverse was not,
and it is the one that drifts unobserved - adding a public method made it
dispatchable-but-unadvertised, and nothing distinguished that from a deliberate
omission.

The 23 capabilities in that gap are now named in `_PYTHON_ONLY_ACTIONS`, so a
new public method fails until it is either published in the enum or recorded as
Python-only. Membership means "dispatchable from Python, deliberately not
advertised to a model", not "must stay unadvertised": whether these should be
published, or the router should instead refuse a non-enum action, is the open
decision in #2093 and is not settled here. No behaviour changes.
