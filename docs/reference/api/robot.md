# Robot and factory

`Robot(name, mode=...)` is a factory function, not a class. It returns a simulation engine in `mode="sim"` (the default) or a `strands_robots.hardware_robot.Robot` in `mode="real"`. Both expose the same agent-facing verbs — `act`, `observe`, `run_policy`, `cleanup` — but two of them accept different first positional arguments in each mode: `run_policy(robot_name=..., policy_provider=..., policy_config=...)` in sim (a factory of policies), `run_policy(policy_object, ...)` on real hardware (a pre-built policy). See the [sim signature](#simulation-run-policy) and the [hardware signature](#the-hardware-robot). After this page you know every keyword the factory accepts and every method the returned object has.

## The factory

::: strands_robots.robot.Robot
    options:
      show_root_heading: true
      heading_level: 3

## The hardware robot

Returned by `Robot(name, mode="real")`. Also usable directly when you already hold a driver.

::: strands_robots.hardware_robot.Robot
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      filters: ["!^_"]

::: strands_robots.hardware_robot.TaskStatus
    options:
      heading_level: 3
      show_root_heading: true

::: strands_robots.hardware_robot.RobotTaskState
    options:
      heading_level: 3
      show_root_heading: true

## Teleoperator

::: strands_robots.teleoperator.Teleoperator
    options:
      show_root_heading: true
      heading_level: 3
