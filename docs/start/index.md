---
description: "The ladder: six rungs from watching a captured agent run to a fleet with one e-stop, each ending with something you can show."
cta: true
copy_prompt: true
---

# Start

Six rungs. Each page ends with a checkpoint: what you now have, and every `python` fence ran against this commit on a laptop with no GPU. Fences needing an arm on USB are noted in the text, not run.

| rung | page | time | you need | you leave with |
|---|---|---|---|---|
| 0 | [See it](see-it.md) | 30 s | nothing | a captured run: an agent builds a scene, runs a policy, and is refused and then approved at the gate in front of a real arm |
| 1 | [Install](install.md), [Run it](first-robot.md) | 3 min | Python `{{extras:python}}` | an SO-101 in MuJoCo: joints moved, state read, a frame saved, a cube on the table |
| 2 | [Talk to it](first-agent.md) | 10 min | a model provider | `Agent(tools=[robot])`, a sentence that moves the sim arm, the operator gate stopping a real one |
| 3 | [Real arm](first-real-arm.md) | 30 min | an SO-101 on USB | the port found, the driver rehearsed on the model, the two lines that move it, calibration, what is refused |
| 3 | [Same checkpoint](first-policy.md) | 15 min | the sim, optionally the arm | SmolVLA from the Hub driving the sim arm from three cameras, and the same `run_policy` call for the real one |
| 4 | [Teach it](teach-it.md) | a day | a GPU for training | a dataset recorded on the robot, a checkpoint trained from it, the checkpoint running back on the robot |
| 5 | [Fleet](fleet.md) | a week | two machines | several robots on the mesh, one dashboard, one e-stop |
| | [Doctor](doctor.md) | 2 min | | `strands-robots doctor`: what each of the fifteen probes checks and what its verdict means |

Rungs 4 and 5 are itineraries through the guides that hold them; every guide page they point at carries fences that ran against this commit.
