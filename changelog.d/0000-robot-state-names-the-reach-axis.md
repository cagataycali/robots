### `get_robot_state` says which way the arm extends

The `end_effector` line now carries the base position, the end effector's
offset from it and the horizontal world axis that offset lies along - `from
base [0, 0, 0]: [+0.02, -0.38, +0.26] (the arm currently extends along -Y)`
for a fresh SO-101 - and the `json` payload the same facts as
`end_effector.base`, `from_base` and `extends_along` (`null` when the arm is
over its base). Before, the position was the only clue to the robot's front:
an agent read "in front of the base" as +X and placed the cube beside an arm
whose whole reach lies along -Y (v0.5.2 devx replay). The offset is measured
from the live floating-base pose when there is one, else the spawn pose.
