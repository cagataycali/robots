---
description: Which object Robot(...) returns in each mode, which module owns what, the seven layers, and the numbers behind them.
---

# Architecture

`strands_robots` is one factory over two lanes and seven import layers. After this page you know which object `Robot(...)` hands you in each mode, which module owns what, the layer rule a change must obey, and the numbers behind it, generated from the tree at build time.

## Two lanes behind one factory

{{drawing:d01_what_is}}

```python title="sketch"
from strands_robots import Robot

sim = Robot("so101")                                    # mode="sim": a MuJoCoSimEngine
arm = Robot("so101", mode="real", port="/dev/ttyACM0")  # lerobot RobotConfig behind hardware_robot.Robot
arm = Robot("so101", mode="real", driver="strands", port="/dev/ttyACM0")  # a native driver from strands_robots.drivers
```

`Robot` in `strands_robots/robot.py` is a function: it resolves the alias through the registry, then dispatches on `mode`:

| mode | returns | built by |
|---|---|---|
| `sim` (default) | a `SimEngine` from the backend registry (`mujoco` default, `newton`, plugin `isaac`) | `simulation.factory.create_simulation` |
| `real`, `driver="lerobot"` (what `auto` resolves to without a registry `hardware.driver`) | `hardware_robot.Robot`, a lerobot `RobotConfig` under a Strands `AgentTool` | `hardware_robot.py` |
| `real`, `driver="strands"` | a class satisfying the `drivers.base.HardwareDriver` protocol, no lerobot import | `drivers.registry` |
| `auto` | `STRANDS_ROBOT_MODE`, else probes USB for a servo controller, else `sim` | `_auto_detect_mode` |

Both lanes are `AgentTool`s, so `Agent(tools=[robot])` works the same on each, and both take a `Policy` from `policies.create_policy` for `run_policy`. Sim is the default so a script never moves hardware by accident.

The sim lane is the `MuJoCoSimEngine` class: `SimEngine` plus mixins for physics, rendering, recording, randomization, manipulation, motion primitives and teleop, exposed as one tool with an `action` vocabulary. The hardware lane is `hardware_robot.Robot` plus the driver layer; a task runs in a background thread with `TaskStatus` and a stop flag, and a policy dispatch passes the operator gate in `_command_gate.py`.

## Layers

{{drawing:d02_layers}}

Seven layers, top to bottom; a module imports only from layers below its own:

```text
core -> registry -> drivers|mesh -> sim|policies -> app -> tools -> dashboard
```

`scripts/check_import_layers.py` grades this from the source with `ast`: no runtime import cycle, and no upward edge unless it is written in the script's `KNOWN_DEFERRED_UPWARD_EDGES` roster. The roster is a ratchet: removing an inversion deletes its line, adding one fails the check until it is listed. It is empty at this commit.

The table is generated from the grader's `LAYERS` declaration and the tree's line counts.

{{module_map}}

Placements that are a judgement, not a reading: `assets` sits with `registry` because it resolves the paths the registry declares; the dataset modules (`dataset_recorder`, `dataset_metadata`, `dataset_source`, `streaming_dataset`, `dataset_transfer`) sit in `core` because a dataset is a contract two layers agree on, not a host.

## Rules every module obeys

- **Cheap import.** `import strands_robots` leaves numpy, torch, mujoco and lerobot out of `sys.modules`; every heavy name in `__all__` is behind the package `__getattr__`, and the import-time shims (`_mujoco_gl`, `_dyld`) are stdlib-only leaves.
- **Registry is the source of truth.** `registry/robots.json` holds {{n:robots}} robots in {{n:categories}} categories with {{n:aliases}} aliases; `policies.json` holds the providers. Code reads the row; it never hard-codes a robot.
- **Refuse, do not guess.** A value the code cannot honour (a non-finite pose, an unknown joint name, a hardware kwarg on a sim robot) is refused with a message naming the valid set; continuable refusals carry a [code](../reference/refusal-codes.md).
- **A policy does not reach hardware without an answer.** Mutative verbs on `use_unitree`, `serial_tool`, `pose_tool`, the `Robot` tool's `execute` and `start`, and every ROS 2 transport raise the SDK interrupt; `STRANDS_*_COMMAND_ALLOW` pre-approves one command for an unattended run. Stop verbs are never gated; the native drivers' `move_to` is not gated yet.
- **Providers are plugins.** Simulation backends register through `register_backend` or the `strands_robots.backends` entry-point group; policies through `register_policy` or `policies.json`; native drivers through `register_native_driver`.

## Extras

The package installs with no heavy dependency; each lane pulls its own extra: `[sim-mujoco]` for the sim lane, `[lerobot]` for the lerobot hardware lane, `[mesh]` for Zenoh, `[dashboard]` for the operator UI, one extra per native driver or policy provider ({{n:policy_providers}} providers, {{n:native_drivers}} shipped drivers). `pyproject.toml` is the list; a door that needs an extra you lack refuses with the install line.

## What changes in 1.0

1.0 keeps this layer DAG and changes the layers' size and the number of contracts ([roadmap](../reference/project/roadmap.md)).
