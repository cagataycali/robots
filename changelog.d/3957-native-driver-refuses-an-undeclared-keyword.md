### Fixed: a native driver accepted a keyword it never read

`Robot(name, mode="real", driver="strands", **kwargs)` forwarded the caller's
keywords into each driver's `**kwargs` sink. Three drivers parked the remainder
in `self._extras`, ten logged a debug line and dropped it, and nothing in the
package read either, so a typo configured nothing and reported success:
`Robot("so101", mode="real", driver="strands", prot="/dev/ttyACM0",
trasport="twin")` built a `FeetechDriver` reporting `port=None` and
`transport="serial"` - a serial arm auto-detecting a port while the caller had
named one, and a twin transport that never happened. The same two typos on
`driver="lerobot"` are refused by name against the robot's config dataclass, so
the native path was the exception.

The keyword roster is now the driver's own signature. `constructor_keywords()`
on `strands_robots.drivers.base` reads it off the class, and the factory refuses
anything outside it in the lerobot path's message shape - naming the keyword and
the roster it is missing from. The sixteen keywords three drivers read off
`kwargs` with `kwargs.pop(...)` (`FeetechDriver`'s nine, `DynamixelDriver`'s
four, `FrankaDriver`'s three) are declared parameters, so `help(FeetechDriver)`
now shows `baud_rate`, `calibration`, `timeout` and `transport` instead of
`**kwargs`; the ten drivers that only discarded their sink no longer declare
one. No shipped driver keeps a sink, pinned over all thirteen classes so the
hole cannot reopen.

Two call sites turned out to pass a keyword their driver never read, which is
the class of defect the tolerance hid: `ReachyDriver(host=...)` in a transport
test - the driver reads the polymorphic `port=` - and a G1 lidar cell passing
`lidar_max_points=`, a parameter an earlier change had already removed.
`tests/test_docs_real_mode_invocations.py` drops its own AST scan of
`kwargs.pop(...)` reads and grades documented calls against the same
`constructor_keywords()` the factory enforces.
