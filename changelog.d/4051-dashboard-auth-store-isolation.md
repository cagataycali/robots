### Changed: the credential store a test reaches is its own, not the machine's

`strands_robots.dashboard.auth._store_path` resolves an unset
`STRANDS_DASH_AUTH_STORE` to `~/.strands_dashboard/auth.json`, and every reader
goes through `_load`, which writes. Sixteen test modules redirected it in an
autouse fixture of their own, no two agreeing on which sibling knobs to unset, so
a module that did not carry the redirect read and wrote the real store. The
redirect and the rest of the `STRANDS_DASH_AUTH_*` family are now a property of
the test session, and a module that declares nothing grades that it is.

Moving the redirect into the session also reordered fixture teardown onto a
latent crash in the shared Device Connect restore: an import a test blocks by
registering `None` in `sys.modules` was read as a module, so the teardown
raised `TypeError: vars() argument must have __dict__ attribute` and reported a
cell whose assertions had passed as an error. A blocked name holds no
attributes and is now skipped.
