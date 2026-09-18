### Fixed: a test that swaps in the real Device Connect extra puts back the modules it displaced

Thirteen test modules bind the genuine `device_connect_edge` by dropping
`strands_robots.device_connect.*` from `sys.modules`, which orphans every
reference a sibling module already holds: the next import hands out a different
object, so a `monkeypatch.setattr` on the sibling's binding is invisible to the
code under test. Four cells in `tests/drivers/test_reachy_wireless_daemon_protocol.py`
resolved a real hostname when the swapping file happened to run first. The swap
and its undoing each have one owner now, and
`tests/test_sys_modules_removal_leaves_no_orphan.py` grades the prefix-purge
idiom its literal-key half could not see.
