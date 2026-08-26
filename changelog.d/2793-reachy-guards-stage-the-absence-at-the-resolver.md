### Fixed: the Reachy transport guards stage the absence at the module the resolver imports

`tests/drivers/test_reachy_transport_guards_are_reachable.py` grades the two
`_resolve_transport` guards that `ReachyDriver.connect_eagerly`'s pre-check
hides, so a stock install is refused by name rather than crashing. It staged
that absence by making `device_connect_edge` unimportable and evicting the
`strands_robots.device_connect` subtree from `sys.modules`, relying on the next
import to re-execute a package `__init__` that fails.

That `__init__` no longer imports the extra - deferring it is the whole point of
the lazy export table it now carries - so the eviction re-executes an `__init__`
that succeeds, the stdlib-only transport leaf imports, and `_resolve_transport`
hands back the real module. The driver then reaches the daemon: six of the
file's nine cells failed with `daemon unreachable (localhost:8000): <urlopen
error [Errno 111] Connection refused>`, a real network call from a unit test.

The eviction had a second cost. Deleting the subtree let the next import build a
fresh module object for `strands_robots.device_connect.reachy_transport`, so a
double installed on one object was not read through the other. Run before its
sibling, this file took `tests/drivers/test_reachy_driver.py` from 109 passed to
46 passed / 72 failed, and the pair from 0.30s to 21.49s of real connection
timeouts.

The absence is now staged at the one module `_resolve_transport` imports, and
nothing else in `sys.modules` is touched. The premise test that should have
caught the drift asserted `from device_connect_edge import` appeared in
`__init__.py`, which the `TYPE_CHECKING` block satisfies - a block the source
itself annotates `never executed`. It now reads the executed module-scope
imports, so a typing-only import no longer reads as an eager one.
