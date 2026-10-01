---
description: "How Strands Robots works in three minutes: one Robot object that is an agent tool, one Policy object that never sees a robot, and the backends under them."
---

# How it works

Strands Robots is one interface to any robot, simulated or physical, that never moves a real arm until a person says yes. Three objects carry the whole design.

{{drawing:d01_what_is}}

## Robot: the execution target

`Robot("so101")` is a factory, not a class. In `mode="sim"`, the default, it returns a simulation engine with the world built and the robot in it; in `mode="real"` it returns the lerobot driver or a native driver for the same name. Both objects are Strands agent tools: they carry a `tool_name`, a `tool_spec` the model reads and one `action` enum, so `Agent(tools=[robot])` works with either. Every method returns the same envelope, `status` plus a `content` list of text, JSON and image blocks, which is what the model sees when it calls the tool and what you print when you call the method. [Robots](robots.md) goes deeper.

## Policy: behaviour as a runtime component

A `Policy` turns an observation and an instruction into a chunk of actions. `create_policy(provider, **config)` builds one from a provider name ({{n:policy_providers}} providers, from `mock` through `lerobot_local` to `remote`) and a configuration, usually a Hub checkpoint. The policy never sees a robot: it sees `observation.state` in the units it was trained on and returns `action` in the same units. The robot object owns the control loop, injects the state keys and the control frequency, and applies each action with `send_action`. [Policies](../learn/policies/index.md) chooses a provider; [Embodiments](embodiments.md) explains the map between the policy's numbers and the robot's joints.

## Backend: where the robot is

Under the same `get_observation`, `send_action` and `run_policy` sits MuJoCo on a CPU, Newton or Isaac on a GPU, the lerobot driver on a USB port, or a native driver speaking a serial bus, DDS or a vendor API. The call does not change; the backend does. [Simulation and hardware](backends.md) lists them and what each refuses.

## The gate

Anything that moves a physical robot passes `gate_motion` first: an allowlist variable, then `BYPASS_TOOL_CONSENT`, then the operator through a Strands interrupt, and with nobody to ask the call fails closed. Every decision is written to the audit log. [Agents and robots](../learn/agents.md) draws the chain and names the one native command that skips it today.

## Around the spine

A [fleet](../learn/mesh/index.md) puts many such robots on one mesh with one e-stop; the [data flywheel](data-flywheel.md) records what a robot did, trains a checkpoint from it and runs the checkpoint back on the robot; [remote inference](../learn/policies/remote.md) moves the policy to a GPU host while the robot host keeps the control loop and the gate. The [glossary](glossary.md) defines the words these pages use.
