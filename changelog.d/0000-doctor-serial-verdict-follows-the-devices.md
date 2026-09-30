### Fixed: `doctor` no longer fails a host that has no serial device

On a machine with no `/dev/ttyACM*` or `/dev/ttyUSB*`, such as a cloud GPU box,
CI or a sim-only laptop, the serial row was `FAIL` whenever the user was not in
`dialout`. So `python -m strands_robots doctor` exited 1 and printed "Some checks
failed" although nothing it could use was broken. That row is now a `WARN` that
names the `usermod` to run before plugging an arm in. With devices connected,
every device is checked (it used to be only the first). A device this process
can open read/write passes however access was granted: the `dialout` group, a
udev rule or an ACL.
