### Added: `Simulation`, one world for many robots, on MuJoCo

`Simulation.default()` is the world every robot joins when it is given none: robots get unique
ids, a control step is `timestep * substeps` for the whole world, `manual_time()` hands stepping
to the caller, `set_pose` teleports (a fixed base is rebuilt in place with its joint state), and
`render()` frames every robot. A key naming a tendon or site actuator (the Panda hand) reads the control that holds it. The `Backend` contract has one implementation, `mujoco` (3.3.4 or newer).
