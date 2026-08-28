### Added: native Feetech driver stub so `Robot("so101", mode="real", driver="strands")` builds

The driver seam (#353 / #2734) has been on `main` for a fortnight with no native driver
registered against any Feetech robot -- `Robot("so101", mode="real", driver="strands")`
raised `ValueError` from `_build_native_driver` because `get_native_driver_class` returned
`None`. The Feetech codec has been in-tree since the protocol PR landed, but no class
satisfied `HardwareDriver` for these arms, so the driver seam refused them by name.

This PR ships the smallest driver package that removes the failure: `FeetechDriver` in
`strands_robots/drivers/feetech/driver.py`. Construction succeeds; the four deferred
verbs (`send_action`, `start_task`, `run_policy`, plus the `move_to`/`set_torque`/`home`
side of `stream`) return the same `{"status": "error", "content": [{"text": "not wired
yet (the Feetech SCS serial bus)"}]}` envelope a caller who plumbed error handling for
the sibling `DynamixelDriver` already handles. Three read-only verbs land now:
`status`, `sensors`, `stop`. The bus that would carry the writes is #360 scope 1 and is
deliberately its own PR -- landable-without-hardware is the whole point.

`get_status` returns a well-formed envelope matching `DynamixelDriver.get_status`'s
shape, so the mesh publishes both peers identically. `connect_eagerly` names the
"not wired" reason rather than returning `None` (which the caller would read as
success) or raising (indistinguishable from a real hardware failure). `cleanup` and
`stop` are honest no-ops because the driver holds no OS resources yet.

Registered for `so100`, `so101`, `lekiwi`, `moss`, `hope_jr`, `open_duck_mini` -- every
Feetech-servo robot #360 names, and no other. Registering for an arm we cannot verify
would be a promise this driver does not yet keep.
