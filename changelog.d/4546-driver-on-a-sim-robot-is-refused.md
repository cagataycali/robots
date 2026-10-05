### Fixed: `Robot(name, driver="lerobot")` on a simulation is refused, not ignored

`driver=` picks a hardware driver, which a simulation does not have. On a sim
robot (`mode="sim"`, the default, or `mode="auto"` with no servo bus) any value
but `"auto"` was logged at debug level and a simulator was built under
`status="success"`, so `Robot("so101", driver="lerobot")` copied from a robot
page without its `mode="real"` left the arm on the desk still. It is now refused
with the `TypeError` that `port=` already gets, naming the mode and the
`mode="real"` remedy.
