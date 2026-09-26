### Fixed: the hardware catalog lists every robot a driver can build

`examples/registry/lerobot_hardware_catalog.py` walked `list_robots(mode="real")`,
which reads the registry's `hardware` block, and published the result as the
robots with hardware support. That left out the 11 robots a shipped native driver
builds without a declaration - `panda`, `fr3`, `fr3_v2`, `ur5e`, `ur10e`,
`vx300s`, `wx250s`, `aloha`, `dynamixel_2r`, `trossen_wxai`, `open_duck_mini` -
and printed `?` as the lerobot type of the 8 listed robots that have none, which
a reader could not tell from a registry defect. It now walks
`list_driver_coverage()`, the join of both registries, and names the driver that
builds each row: 35 robots, none unresolved.

Its `--g1` mode closed by printing `Robot('g1', mode='real',
robot_ip='192.168.123.164', controller='GrootLocomotionController')`. The factory
grades a native driver's keywords against its own signature, so that recipe is
`ValueError: Unknown kwarg(s) for 'unitree_g1' on driver='strands':
['controller', 'robot_ip']`; `port=` is the spelling the G1 bring-up page
documents.

`docs/getting-started/robot-factory.md` taught the join with `coverage["panda"]`
as the empty tuple that is the driver gap, which the Franka driver made
`('strands',)`. `format_robot_table()` and `list_robots()` now say that their
`Real` column and `"real"` filter read a declaration, and name
`list_driver_coverage()` as the complete answer.
