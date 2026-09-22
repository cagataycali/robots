---
description: SimEngine(ros2_bridge=True) - publish a running simulation's joint_states and camera image_raw on a ROS 2 domain so rviz, nav2 and agents can subscribe.
---

# Sim bridge: publish a simulation on a ROS 2 domain


The simulator can advertise its own live state on ROS 2. Construct any
`SimEngine` (e.g. `Simulation()`) with `ros2_bridge=True` and it spins up an
internal `rclpy` node that publishes, per robot, after every `step()`:

| Topic | Type | Content |
|-------|------|---------|
| `/<robot>/joint_states` | `sensor_msgs/msg/JointState` | joint names + positions |
| `/<robot>/<camera>/image_raw` | `sensor_msgs/msg/Image` (`rgb8`) | one frame per attached camera. `<robot>`/`<camera>` are sanitised into ROS 2 name tokens, so a camera named `0` publishes on `/<robot>/camera_0/image_raw` - ROS 2 forbids a token starting with a digit |

`name` and `position` are one table read by index. A robot whose observation
does not carry every joint of `robot_joint_names()` - every floating-base
humanoid, quadruped and mobile base, whose root freejoint is joint 0 and is not
an observation key - publishes only the joints it observed, so the two arrays
stay the same joints. A caller that hands `publish_joint_states` a differing
number of names and positions has no pose to publish: the message is dropped
whole with a warning naming both counts, rather than published with every joint
after the gap under its neighbour's name.

```python
from strands_robots.simulation import Simulation

sim = Simulation(ros2_bridge=True, ros2_domain=0)
sim.create_world()
sim.add_robot("so101")
sim.step(10)   # publishes /so101/joint_states (+ camera image_raw) on domain 0
```

External ROS 2 nodes - and the agent's own `use_ros` calls - then see the
running simulation:

```bash
ros2 topic list | grep so101          # /so101/joint_states, /so101/<cam>/image_raw
ros2 topic echo /so101/joint_states   # live joint positions, updated every step
```

`rclpy` is an optional, system-provided dependency: it arrives with a sourced ROS
2 distro, not with the `[ros2]` extra (which installs only the cyclonedds RMW
binding, as above). When it is missing, `ros2_bridge=True` raises an `ImportError`
at construction naming the `source /opt/ros/<distro>/setup.bash` step that
supplies it; `ros2_bridge=False` (the default) never touches ROS 2, so the base
sim install stays lightweight. The bridge node is torn down cleanly on `destroy()`.

See `examples/ros2/sim_bridge_demo.py` for a runnable end-to-end script.

