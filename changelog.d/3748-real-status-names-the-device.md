### Fixed: a real robot's `status` names the device, and an idle arm is not an error

`Robot(..., mode="real")` `status` now reports, under the task state, whether
the device is connected, whether its port is present on this host (an absent
port is named, with the remedy, instead of reading identically to a healthy
arm at rest), and each configured camera's state - as text and as a `json`
block. Presence is only reported where it was read: a network port such as
Reachy 2's `50065` is named as a network port reached at its address, not
claimed present on a host that has no such path to check. `get_status()` no
longer reports every not-yet-connected arm as an error: `is_calibrated` is `None` until the bus is open rather than a raise.
