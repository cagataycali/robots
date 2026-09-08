### Fixed: the VERA docker readiness wait can reach its own deadline

`DockerServerRunner._wait_until_ready` polls `while time.monotonic() < deadline`
and made two blocking calls one line apart inside that loop: `_port_open`, bounded
at 1 s, and `_container_running`, which ran `docker ps` with no bound. A docker
client whose daemon has stopped answering blocks in that read, so the deadline was
never re-evaluated - the wait that documents "or raise on timeout" could not reach
its timeout, and because the raise is what calls `stop()`, the container it had
just launched was never torn down. Measured with `server_ready_timeout=3.0` (a
value every check on the config accepts) and an unresponsive daemon, the wait was
still blocked at ten times the budget, having probed the port zero times.

Every `docker` *query* now states the same 10 s bound `docker logs` already used,
and a query that does not answer raises `docker did not answer 'ps' for container
... within 10s, so whether it is running is unknown` and tears the container down.
That is its own cause rather than the "exited before becoming ready" the wait
reported from a `False` it never received. `docker run` is not a query and stays
unbounded: it may pull the image, and it runs before any readiness budget starts.
