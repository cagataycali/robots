### Tests: twelve more domain sweeps walk their value tables in one cell

The streaming-open knob, bucket publication flag, bridge read timeout,
robot_mesh numeric option, randomize axis flag, panorama and splat background,
Unitree LowCmd field, predicate tolerance and finiteness, colour, registry
name and ROS DDS credential suites gave every probe value its own cell. Each
test now walks its table in one cell and names the failing value in the
assertion. The twelve files go from 1,405 cells to 520 with the same package
lines executed.
