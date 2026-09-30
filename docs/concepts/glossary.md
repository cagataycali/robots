---
description: "The words these docs use with a fixed meaning: action, backend, checkpoint, embodiment, envelope, extra, gate, grader, lane, policy, provider, refusal, rollout, rung, sketch, transport."
---

# Glossary

**action** · One step of robot command: a dict of joint or motor targets, or a list of them (a chunk) when the policy emits several per inference. Also the `action` field an agent tool call carries, naming which of the tool's verbs to run.

**backend** · Where a robot object runs: a simulator (MuJoCo, Newton, Isaac) or a hardware driver (lerobot or native). The interface is the same; the backend is what `mode=` and `backend=` or `driver=` select. [Simulation and hardware](backends.md).

**checkpoint** · The weights of a trained model, on the Hub or on disk, named by `pretrained_name_or_path`. A checkpoint plus an embodiment plus a processor is a policy.

**embodiment** · The declared map between a policy's tensor slots and units and a robot's keys and units. One robot can have a sim embodiment and a real one. [Embodiments](embodiments.md).

**envelope** · The dict every robot action returns: `status` and a `content` list of `text`, `json` and `image` blocks. What the agent reads and what you print.

**e-stop** · The command that latches every robot on a mesh or in a dashboard into a stopped state until an operator resumes it with proof. Three ways to raise it, two ways to resume; [Fleet](../learn/mesh/fleet.md).

**extra** · An optional dependency group: `pip install "strands-robots[sim-mujoco]"`. A feature whose extra is missing refuses by naming the extra.

**fence** · A code block on these pages. Bare `python` fences run against the commit in CI; `python title="sketch"` fences open a serial port or need hardware and are verified statically.

**gate** · `gate_motion`: the check every command that can move a physical robot passes, in a fixed order, ending in an operator interrupt or a refusal. [Agents and robots](../learn/agents.md).

**grader** · A test under `tests/test_docs_*.py` that reads the docs and the code together and fails when they disagree: a fence without output, a page over its word cap, a claim the code does not keep.

**interrupt** · The Strands mechanism that pauses an agent turn to ask a person. The gate raises one named `<tool>-command-approval`; `y` resumes and dispatches, anything else declines.

**mesh** · The Zenoh network robots and dashboards join to see each other, exchange commands and share one e-stop; postures from mTLS to local development. [Mesh](../learn/mesh/index.md).

**policy** · The runtime object that maps an observation and an instruction to actions. Learned (a checkpoint), planned (a motion planner) or scripted (`mock`). [Policies](../learn/policies/index.md).

**provider** · A named way of building a policy or a trainer: `lerobot_local`, `remote`, `wbc`, `mock`. `create_policy(provider, ...)`.

**refusal** · An error envelope that names what was wrong and what would have been accepted. The docs treat a refusal as an answer; [refusal codes](../reference/refusal-codes.md).

**rollout** · One run of a policy on a robot for a duration or a number of steps: `run_policy`, `execute`, `start_task`.

**rung** · A stage of the [ladder](../start/index.md): See it, Run it, Talk to it, Real arm, Teach it, Fleet.

**sketch** · See fence.

**transport** · What carries a native driver's bytes to the motors: a serial port, a TCP socket, DDS, ROS 2. Named on every hardware robot page.

**twin** · A MuJoCo model of a physical arm stepped by the same driver, `transport="twin"`, so a command can be rehearsed before the arm moves.
