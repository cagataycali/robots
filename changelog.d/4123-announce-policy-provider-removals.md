### Deprecated: `curobo`, `moveit2`, `kimodo`, `protomotions` and in-process GR00T are removed in 0.7

`create_policy` now raises a `DeprecationWarning` for each of these four
providers, naming what replaces it, and `Gr00tPolicy(model_path=...)` does the
same for in-process GR00T. Each provider page carries the notice.

| removed in 0.7 | use instead |
|---|---|
| `curobo` | `simulation.motion_primitives` with mink IK for a sim reach, or Isaac cuMotion for GPU planning |
| `moveit2` | a MoveIt goal sent as a ROS 2 action through the `use_ros` or `use_rosbridge` tool |
| `kimodo` | a motion generated offline and replayed as joint targets (nothing in-tree) |
| `protomotions` | the `wbc` provider for Unitree G1 whole-body control |
| `Gr00tPolicy(model_path=)` | service mode (`host=`/`port=`), or `create_policy("lerobot_local", policy_type="groot", pretrained_name_or_path=...)` |

Nothing else changes in 0.6: each provider still constructs and runs.
