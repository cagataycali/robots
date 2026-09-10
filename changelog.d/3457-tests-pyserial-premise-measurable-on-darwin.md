### Fixed: the pyserial-coercion grader measures on macOS too

`tests/test_bus_speed_is_graded_wherever_a_port_is_opened.py::TestPyserialCoerces`
opened a pty at 1, 2 and 1_000_000 baud to show pyserial applies a coerced
speed. macOS sets a non-standard rate through the `IOSSIOSPEED` ioctl, which a
pty refuses with `ENOTTY`, so three of the four cases failed on every Mac
checkout of `main`. The coercion half is now measured on an unopened `Serial`
(every OS), and the applied half against the pty at 9600 baud.
