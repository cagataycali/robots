### Fixed: the sim tool description tells the agent about the world it is joining

`Robot("so101")` creates the world and adds the robot before the agent sees the
tool, but the tool description was a constant that said the session starts with
`create_world`. An agent following it had its first call refused - `a world
already exists` - in 8 of 8 measured sessions across six embodiments, and had
no joint names to work from.

The opening sentence is now written from the live world: with robots loaded it
names them and their joints (first 8), says `create_world` is refused while a
world exists, and points at `get_robot_state` / `set_joint_positions` /
`move_to` / `step` / `render` / `reset`; with no world it keeps the
state-machine sentence, which is then true. Measured with the same prompts:
the SO-101 hello session went from 8 calls with 2 errors to 3 calls with none.

The offered actions follow each robot's resolved actuator ownership. A model
that compiles with no actuator can be posed, stepped and rendered, but `move_to`
refuses it outright and `run_policy` can only advance physics without commanding
it, so such a session is pointed at `actuate_robot` - the same remedy
`add_robot` and `move_to` already name - and the robot is marked
`[no actuators]`. A world holding one actuated robot keeps the drive verbs.
