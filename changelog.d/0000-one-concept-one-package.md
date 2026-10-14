### Changed: one concept, one package (part 3.5)

Each of the five types is now a package you can read top to bottom -- `simulation/` (with
`simulation/backends/mujoco`), `robot/` (with `robot/drivers/` and the one run loop in `robot/run/`),
`policy/`, `description/`, `units/` -- and a type lives with its owner. The top level exports only
`Robot`, `Simulation`, `Policy`, `Body`, `Description` and the four data types; everything else is
imported from its package. Facts about a robot (its gripper keys, LeRobot norm modes, its driver) are
registry data, so the `lerobot_so10x` units become the `lerobot` scheme read from
`hardware.lerobot_norm`. No file exceeds 250 lines and `import-linter` contracts guard the edges.
Behaviour is unchanged; a new robot is a registry edit, not a code edit.
