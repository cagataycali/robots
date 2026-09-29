### Fixed: a service policy connects without a websockets `DeprecationWarning`

`RemotePolicy` and the Cosmos 3 transport each hold one `websockets` connection
across every request, obtained with a bare `connect(...)`. websockets 17.1
deprecates that spelling (a `DeprecationWarning` on the connection's first
read, and `connect()` is announced to change behaviour when the period ends),
so a rollout run with `-W error::DeprecationWarning` fell at the first connect.
Both clients now pass `legacy=True`, the supported way to hold a connection
outside a `with` block; the flag arrives in 17.1, so the `[cosmos3-service]`
and `[inference]` floors move from `websockets>=17.0` to `>=17.1` (#4187).
