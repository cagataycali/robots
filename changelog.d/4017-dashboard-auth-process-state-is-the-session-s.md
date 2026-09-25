### Fixed: the dashboard-auth process state is the test session's to restore

`strands_robots.dashboard.auth` keeps three process-globals -- the parsed
credential store, the ceremonies awaiting a finish, and the diagnosis of a store
that would not parse -- and all three outlive the test that filled them, so a
store one test wrote answered a later test's read and a ceremony one test
stashed counted against a later test's per-ip cap. Fourteen of the fifteen
`tests/test_dashboard_auth_*` modules reset some of it in a fixture of their own
and no two reset the same set; measured over the six that touch the store, every
one handed the next module a populated store cache and two handed on two pending
challenges. `tests/conftest.py` restores it now, beside the four process-globals
the session already owned, and the three one-per-global boundary pins become one
table with a row per global and a roster cell that fails when a fourth appears.
