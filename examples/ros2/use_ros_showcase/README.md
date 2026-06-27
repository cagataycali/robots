# `use_ros` live showcase

A reproducible, one-command demonstration of the `use_ros` tool driving a
**real ROS 2 `turtlesim` node** entirely in-process through `rclpy` - no `ros2`
CLI, no code-generation. It exercises every action and captures real data,
including a closed **sense -> act -> sense** loop and the structured-error
contracts.

## A Strands agent driving the turtle (natural language -> motion)

`agent_drive.py` hands the `use_ros` tool to a Bedrock-backed Strands agent
(`global.anthropic.claude-opus-4-8`) and asks it, in plain English, to "draw a
square". The agent autonomously issues **9 `use_ros` tool calls** (4 sides, 3
turns, stop, echo) and reports the final pose - all in-process through rclpy.

![A Strands agent drawing a square in turtlesim via use_ros](../../../docs/assets/use_ros_agent_turtle.gif)

```
pose BEFORE: (5.54, 5.54, 0.0)
=== AGENT DRIVING (square) ===   [model: global.anthropic.claude-opus-4-8 via Amazon Bedrock]
Side 1: Tool #1: use_ros   Turn 1: Tool #2: use_ros   ...   Stop: Tool #8   Echo: Tool #9
"Square complete! Final pose: x=5.35, y=6.73, theta~2.84 rad, velocities at zero."
pose AFTER: (5.35, 6.73, 2.84)
```

Full transcript in [`agent_sample_output.txt`](./agent_sample_output.txt); the
recording above is the live turtlesim canvas captured while the agent drove.

```bash
# inside a sourced ROS 2 env with turtlesim running:
export AWS_BEARER_TOKEN_BEDROCK=...   # or any boto3 credential chain
pip install strands-agents
python3 agent_drive.py
```

## Run it

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
