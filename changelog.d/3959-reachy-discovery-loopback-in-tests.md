### Fixed: the driver verb-dispatch grader no longer probes the network for a Reachy daemon

`tests/drivers/test_an_undeclared_verb_is_refused_not_dispatched.py` drives the
real `ReachyDriver.stream` for every verb the schema declares. Without
`REACHY_HOST` set, discovery probed the shipped default list - `localhost`, then
`reachy-mini.local` - so each of the 22 daemon-backed verbs paid the resolver's
~5s timeout for that mDNS name, making it the most expensive cell in the suite
at 110.4s. On a network where the name does resolve the sweep commanded the
robot that answered, `wake` and `body_turn` included. The module now points
discovery at a closed loopback port and grades the candidate list, so the
dispatch assertions are unchanged while the module runs in 0.75s.
