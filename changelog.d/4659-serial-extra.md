### Added: a `[serial]` extra for the Feetech and Dynamixel drivers and the serial tools

`pip install 'strands-robots[serial]'` installs pyserial with the project's
bound. When pyserial is absent, `serial_tool`, `pose_tool`, `FeetechBus.connect()`
and `DynamixelBus.connect()` now name that extra before `pip install pyserial`,
the way every other native driver's refusal names its own extra. `[dashboard]`
reaches pyserial through `[serial]` instead of a second copy of the bound, and
`[all]` includes it.
