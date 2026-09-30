---
description: "One interface, several places a robot can be: MuJoCo, Newton and Isaac in simulation, the lerobot driver and native drivers on hardware, and what each one refuses."
---

# Simulation and hardware

`get_observation`, `send_action`, `run_policy` and the agent tool surface are the interface. What sits under them is a backend, and the backend is the only thing that changes between a laptop and a lab.

{{drawing:d05_driver_stack}}

## Simulation backends

| backend | selects with | runs on | for |
|---|---|---|---|
| MuJoCo | default, `backend="mujoco"` | CPU | one world, one or a few robots, every Start page |
| Newton | `backend="newton"` | GPU (Warp) | many worlds stepped together |
| Isaac Sim | `backend="isaac"`, plugin `strands-robots-sim` | GPU, NVIDIA | photoreal scenes, Isaac assets |

All three build from the same registry entry and expose the same engine methods; `list_backends()` names what this install can start. The simulator is never gated: a sim world is the place where an agent may do anything, which is why the [ladder](../start/index.md) keeps you there until Stage 3.

## Hardware backends

`mode="real"` chooses a driver. `driver="lerobot"` wraps a lerobot robot class (SO-101, Koch, LeKiwi, Reachy 2, Unitree G1 through lerobot's own DDS client) and inherits its calibration files and camera handling. `driver="strands"` picks a native driver from `strands_robots.drivers` ({{n:native_drivers}} of them: Feetech and Dynamixel serial buses, Franka, Robotiq, Unitree, ROS transports), which talks to the bus or the vendor API directly; with `transport="twin"` the same driver steps a MuJoCo model of the arm instead, so a command can be rehearsed before the arm moves. `driver="auto"`, the default, honours the registry entry's `hardware.driver` and otherwise builds the lerobot driver; `"strands"` is refused by name when no native driver exists, never quietly replaced.

A hardware robot publishes a smaller tool: read (`status`, `get_state`, `list_cameras`, `render`), stop, and the gated `execute` and `start`. It refuses a second rollout while one holds the bus, every call after `cleanup()`, and a command whose driver is not installed, naming the extra to install.

## Transports

Between a native driver and the motors sits a transport: a serial port for Feetech and Dynamixel, a TCP socket for Franka and Robotiq, DDS for Unitree, a ROS 2 graph or rosbridge for anything with a ROS stack. The robot page for each hardware robot ([SO-101](../robots/so101.md), say) names its transport and the address the driver needs, and [Real arm](../start/first-real-arm.md) shows the port discovery once.

## The same call, checked

Every guide on this site that shows a sim fence and a real sketch uses the same method names, and a grader reads both to make sure they spell one API. When a backend cannot honour a call it refuses with a sentence naming what it lacks; it does not approximate. `render` on a real arm returns the camera frame; `randomize` sent to a real arm's tool is answered by naming the tool that has that action, the simulator's.
