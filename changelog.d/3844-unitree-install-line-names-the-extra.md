### Fixed: the Unitree SDK refusal installs the DDS binding through `[ros2]`

The recipe every G1/Go2/Booster missing-SDK refusal prints opened with `pip
install 'cyclonedds>=0.10.2,<12'`, a second copy of a range the manifest already
owns: `[ros2]` declares exactly that requirement. A reader following the refusal
therefore pinned a bound no release note would move with, and the driver tree was
back to handing out a distribution instead of a capability. The line now reads
`pip install 'strands-robots[ros2]'` - same wheel, one declaration - in
`strands_robots.drivers.unitree._common` and in
`docs/robots/humanoids.md`. The vendor checkout is unchanged: `unitree_sdk2py`
has no usable PyPI build, so no requirement can carry it.
