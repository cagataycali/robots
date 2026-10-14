### Added: acceptance check 9, a feetech `Driver` on a physical SO-101

`STRANDS_SO101_PORT=<port> pytest tests_integ/acceptance` drives the arm through the v1 surface:
`observe()`, a refused `confirm_act()` that writes nothing, an allowed hold the twin mirrors, and
`calibrate(interactive=False)` against LeRobot's calibration file. Without an arm it skips and says why.
