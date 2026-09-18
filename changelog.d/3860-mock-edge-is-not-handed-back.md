### Fixed: a mock `device_connect_edge` installed at collection time is not handed back by the file that installed it

Two test modules install `MagicMock` stand-ins for `device_connect_edge` at
import time and undo the swap at `teardown_module`. Collection of every file
precedes the teardown of any, so the second file's snapshot was taken while
the first file's stand-ins were resident - the "originals" it recorded were a
`MagicMock` and an integration bound to a class defined in a test file, and its
teardown handed both back to every later import. And the first file's snapshot
predated a real module a later-collected sibling imported, so putting the
snapshot back dropped that module and orphaned the sibling's binding: four
cells in `tests/drivers/test_reachy_wireless_daemon_protocol.py` resolved
`reachy-a.local` for real when selected after it. The one owner of the swap,
`tests/_device_connect_real.py`, now reads a snapshot by what it holds: a real
module goes back, a module bound to a fake does not, a newer real module is
left where the sibling that imported it can reach it, and the stand-ins are
installed and taken away by the same owner.
