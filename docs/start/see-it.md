---
description: "Thirty seconds: one recorded conversation. An agent builds a simulated scene from a sentence, runs a policy, and must ask a person before driving the real arm."
---

# See it

Nothing to install. This is one recorded conversation with a Strands Agent that has two tools: a simulated SO-101 arm (`so101_sim`) and the real one (`so101`), unplugged. Lines starting with `>` are what the person typed; the rest is what the agent did.

{{sim:talk-to-it-1|before the first sentence: the simulated arm alone on its table}}

## Build a scene from a sentence


```text
> In the simulation, add a camera named 'front' that looks at the so101 from the front, then add a red cube 5 cm wide named 'red_cube' on the table 20 cm in front of the arm base, and render the front camera.
tool_use so101_sim {"action": "get_robot_state", "robot_name": "so101"}
tool_use so101_sim {"action": "add_camera", "name": "front", "position": [0.0, -0.8, 0.4], "target": [0.0, -0.2, 0.1], "fov": 50}
tool_use so101_sim {"action": "add_object", "name": "red_cube", "shape": "box", "size": [0.05, 0.05, 0.05], "position": [0.0, -0.2, 0.025], "color": [1.0, 0.0, 0.0], "mass": 0.1}
tool_result success 'red_cube' added: box at [0.0, -0.2, 0.025], size=[0.05, 0.05, 0.05], 0.1kg
tool_use so101_sim {"action": "render", "camera_name": "front", "width": 1280, "height": 720}
```

{{sim:talk-to-it-2|what the agent built: a front camera, and the red cube in front of the arm}}

Nobody told the agent which way the arm faces: it read the state first, saw that the front is -Y, and put the cube there.

## Run a policy on it


```text
> Run the mock policy in the simulation with the instruction 'pick up the red cube' for 3 seconds.
tool_use so101_sim {"action": "run_policy", "policy_provider": "mock", "instruction": "pick up the red cube", "duration": 3}
tool_result success Policy complete on 'so101' | MockPolicy | 3.0s | 150 steps
  Note: MockPolicy does not read the instruction. Its actions - a test motion on every joint - were commanded to the robot whatever the task says; nothing above means the task was performed.
```

{{sim:talk-to-it-3|after the run: every joint moved, and the cube is still where it was}}

No approval so far: nothing in a simulation can get hurt.

## Ask for the real arm

Now the same request, to the real arm:

```text
> Now run the same mock policy on the real so101 for 2 seconds with the same instruction.
tool_use so101 {"action": "execute", "instruction": "pick up the red cube", "policy_provider": "mock", "duration": 2}
interrupt robot-command-approval: 'execute' drives the real robot 'so101' for up to 2s with 'pick up the red cube' (policy mock built in this process, no server); it needs operator approval before it is dispatched. [...] Reply 'y' to approve, anything else to deny.
operator> n
tool_result error so101: execute to 'so101' was declined by the operator.
> Try the real so101 once more, same policy, same instruction, 2 seconds.
interrupt robot-command-approval: 'execute' drives the real robot 'so101' for up to 2s [...]
operator> y
tool_result error Robot connection failed: Could not connect on port '/dev/null'.
```

The agent cannot move the real arm on its own. On no, it was told and could explain why. On yes, the command went out and failed with a plain error: no arm was plugged in. The full conversation is in [talk-to-it.txt](../assets/transcripts/talk-to-it.txt).


That is the whole idea: a robot is a tool, an agent builds and drives a simulated scene from words, and a real arm moves only after a person says yes. To build it yourself, [Install](install.md), then [Run it](first-robot.md).
