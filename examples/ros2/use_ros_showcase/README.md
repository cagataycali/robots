# `use_ros` live showcase

A reproducible, one-command demonstration of the `use_ros` tool driving a
**real ROS 2 `turtlesim` node** entirely in-process through `rclpy` - no `ros2`
CLI, no code-generation. It exercises every action and captures real data,
including a closed **sense -> act -> sense** loop and the structured-error
contracts.

## Run it

```bash
cd examples/ros2/use_ros_showcase
docker compose run --build --rm showcase
```

Builds `ros:jazzy` + `turtlesim` + `strands-agents`, starts a real turtlesim
node, and runs `showcase.py` against it. Exits `0` iff the turtle moved.

## What it proves

| Action | What the live run shows |
|--------|-------------------------|
| `status` | `backend: rclpy (in-process)` |
| `list_topics` / `list_nodes` / `list_services` | the real graph - including our own `/strands_robots_use_ros` node |
| `info` | live type + publisher/subscriber counts for `/turtle1/cmd_vel` |
| `echo` | real `turtlesim/msg/Pose` samples as JSON |
| `publish` | a `geometry_msgs/msg/Twist` that **moves the turtle** (velocities latch to the exact values sent) |
| `service_call` | `/spawn` returns `{"name": "t2"}`, and `t2`'s topics then appear in `list_topics` |
| error: bad type | `nonexistent_pkg/msg/Foo` -> `{"status": "error"}` (`No module named 'nonexistent_pkg'`), never a crash |
| error: bad name | `/bad; rm -rf` rejected by input validation |

The headline: a closed loop driven purely by `use_ros` -

```
before: (x=5.544, y=5.544, theta=0.000)
publish Twist {linear.x=2.0, angular.z=1.8} x20
after:  (x=4.428, y=6.500, theta=-1.445)   linear_velocity=2.0  angular_velocity=1.8
```

A captured run is saved in [`sample_output.txt`](./sample_output.txt).

## Files

| File | Role |
|------|------|
| `showcase.py` | Exercises every `use_ros` action; asserts the turtle moved. |
| `run_showcase.sh` | Starts a headless turtlesim, then runs `showcase.py`. |
| `Dockerfile` | `ros:jazzy` (provides rclpy) + turtlesim + `strands-agents`. |
| `docker-compose.yml` | One-command build + run. |
| `sample_output.txt` | A captured live run for reference. |

## Notes

- `use_ros` needs `rclpy` importable, which means a **sourced ROS 2 distro**
  (rclpy / `rosidl_runtime_py` are not on PyPI). The `ros:jazzy` base image
  provides it; only `strands-agents` is pip-installed. To run outside Docker,
  `source /opt/ros/<distro>/setup.bash` first.
- Already on a ROS 2 machine? Just run `showcase.py` with turtlesim up - no
  container needed.
