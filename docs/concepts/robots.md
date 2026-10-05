---
description: "What Robot() returns, why it is a factory, what the two modes give you, and why nearly every method returns the envelope an agent reads."
---

# Robots

A robot in Strands Robots is an object that owns a body and a control loop, and that a Strands Agent can call as a tool. The body may be simulated or physical; the object's surface is the same shape either way.

## The factory

`Robot(name, mode="sim" | "real", ...)` looks the name up in the [registry](../robots/index.md) ({{n:robots}} robots, {{n:aliases}} aliases) and returns one of two things:

| mode | returns | backend | tool name |
|---|---|---|---|
| `sim` (default) | a `Simulation` engine with the world created and the robot added | MuJoCo by default, `backend="newton"` or `"isaac"` | `<name>_sim` |
| `real` | the lerobot driver, or the native `HardwareDriver` the registry names (`driver="strands"` insists on it) | a USB port, DDS, a vendor API | `<name>` |

With no `mode`, `STRANDS_ROBOT_MODE` decides, then a hardware probe, then a USB scan, and sim is the fallback, so a laptop with nothing attached always gets a simulator. A hardware keyword on a sim robot (`cameras=`, `driver=`) is refused, not ignored. A misspelt name is refused with the nearest matches.

## One engine, many robots

The simulation object is the engine, not a wrapper around one robot. It holds one world and any number of robots, which is why every method that touches a robot takes `robot_name`. `add_robot`, `add_object`, `add_camera`, `randomize` and `render` work on the world; `send_action`, `get_robot_state` and `run_policy` work on a named robot. [Run it](../start/first-robot.md) walks through this with an SO-101.

## The envelope

Every call but `get_observation()`, `cleanup()` (`None`) and listers `list_robots()`, `robot_joint_names()`, `list_cameras()` (a `list`) returns `{"status": "success" | "error", "content": [...]}`, each block `{"text": ...}`, `{"json": ...}` or `{"image": ...}`. Two consequences: What you print in a script is exactly what the model reads when the same object is a tool, so a fence's output on these pages is also a transcript. And an error is an answer, not an exception: an unknown joint name returns `status="error"` with the valid names, and the model can recover in the next turn.

## Robot, embodiment, registry entry

Three words that are easy to blur. The **registry entry** is the description on disk: name, aliases, model files, driver, cameras. The **robot object** is the running thing `Robot()` returned, sim or real. The **embodiment** is the [map](embodiments.md) a policy uses to speak to that object's joints in the policy's own units; one registry entry can have a sim embodiment and a real one. A checkpoint is chosen for an embodiment, not for a robot.

## What a robot refuses

Sim refuses hardware keywords and unknown names. Hardware refuses a second rollout while one holds the bus, refuses every command after `cleanup()`, and refuses `execute` and `start` until the [gate](../learn/agents.md) says yes. Reading and stopping are never refused. The [refusal codes](../reference/refusal-codes.md) page lists every sentence a refusal can carry.
