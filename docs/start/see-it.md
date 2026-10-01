---
description: "Stage 0, thirty seconds: a captured agent run. It builds a scene in the simulator from a sentence, runs a policy, and is refused and then approved at the operator gate in front of the real arm."
---

# See it

Nothing to install on this page. It is a captured run of `docs/hooks/transcripts/talk_to_it.py`: one Strands Agent holding two tools, the simulated SO-101 (`so101_sim`) and a `mode="real"` SO-101 with nothing on its port (`so101`), every call recorded. The `>` lines are what a person typed; the rest is what came back.

{{sim:talk-to-it-1|the scene before the first sentence: Robot("so101") and nothing else}}

## Build a scene from a sentence


```text
> In the simulation, add a camera named 'front' that looks at the so101 from the front, then add a red cube 5 cm wide named 'red_cube' on the table 20 cm in front of the arm base, and render the front camera.
tool_use so101_sim {"action": "get_robot_state", "robot_name": "so101"}
tool_use so101_sim {"action": "add_camera", "name": "front", "position": [0.0, -0.8, 0.4], "target": [0.0, -0.2, 0.1], "fov": 50}
tool_use so101_sim {"action": "add_object", "name": "red_cube", "shape": "box", "size": [0.05, 0.05, 0.05], "position": [0.0, -0.2, 0.025], "color": [1.0, 0.0, 0.0], "mass": 0.1}
tool_result success 'red_cube' added: box at [0.0, -0.2, 0.025], size=[0.05, 0.05, 0.05], 0.1kg
tool_use so101_sim {"action": "render", "camera_name": "front", "width": 1280, "height": 720}
```

{{sim:talk-to-it-2|what the agent built: the camera and the red cube it placed after reading the base pose}}

The model read the state first, learned the arm extends along -Y, and put the cube at `y = -0.20`.

## Run a policy on it


```text
> Run the mock policy in the simulation with the instruction 'pick up the red cube' for 3 seconds.
tool_use so101_sim {"action": "run_policy", "policy_provider": "mock", "instruction": "pick up the red cube", "duration": 3}
tool_result success Policy complete on 'so101' | MockPolicy | 3.0s | 150 steps
  Note: MockPolicy does not read the instruction. Its actions - a test motion on every joint - were commanded to the robot whatever the task says; nothing above means the task was performed.
```

{{sim:talk-to-it-3|after the rollout: the test motion moved every joint, the cube is where it was}}

Nothing asked for approval so far: the simulation is never gated.

## Ask for the real arm

The real arm is:

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

The decline reached the model as a refusal it could explain; the approval dispatched the rollout, which then failed honestly because no arm was on the port. The full log with every message is [talk-to-it.txt](../assets/transcripts/talk-to-it.txt).


You have now seen the whole shape of the project in one run: a robot is a tool, the model builds and drives a simulated scene from words, and a real arm moves only after a person says yes. Next rung: [Install](install.md), then [Run it](first-robot.md) to build that scene yourself.
