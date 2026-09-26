### Changed: the dashboard-auth cells share one connection stand-in, and it is the real one

`strands_robots.dashboard.auth` decides on three facts about a connection: the
headers it carries, the peer address of its socket, and the scheme it was
reached over. Twelve test modules each carried their own stand-in for that -
nine `FakeRequest` classes over a `headers` dict, two builders of an ASGI scope,
one `_Req`/`_Url` pair - in four spellings of one constructor, two of which
replaced the header mapping rather than merging it, so a cell adding a header
dropped `Host`. Two of the three facts are not headers, so a dict stand-in left
the scheme reading a default and could not hold one header twice at all. They
now share `tests/_dashboard_connection.py`, which builds the real
`starlette.requests.Request` / `WebSocket` from the scope a server hands it;
coverage of `dashboard/auth.py` is unchanged (525 statements, 80 missed, the
same missing lines) and the same 267 cells pass. The repeated-header case a
mapping cannot express is now pinned by
`tests/test_dashboard_auth_reads_a_repeated_forwarding_header.py`: a two-hop
chain sent as the header twice rather than joined is the same evidence of a hop,
and the per-ip challenge cap keys on the client at the far end. Towards #3818.
