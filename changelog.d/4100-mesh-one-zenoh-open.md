### Changed: `get_session` opens Zenoh through the same body the bridge leg uses

`strands_robots.mesh.session.get_session` carried its own copy of the Zenoh
open (port parse, auth-mode stash, auto-listener, client fallback, explicit
endpoints) beside `_get_zenoh_session_directly`, so a fix to one had to be
remembered in the other. `get_session` now answers the kill switch and the
backend branch, then delegates, matching how `release_session` and `put`
already route. No behaviour change.
