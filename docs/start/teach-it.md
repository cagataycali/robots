---
description: "Stage 4, a day: record episodes on your SO-101, verify and label them, train a checkpoint, run it back on the arm, and write down the number it earned."
---

# Teach it

At the end of this rung a checkpoint trained on episodes you recorded runs on the robot that recorded them, in the simulator first and then behind the gate on the arm, and its success rate is written down with the dataset and the checkpoint that produced it. This is one turn of the [data flywheel](../concepts/data-flywheel.md); the pages below are its steps, in order, each with fences that ran against this commit.

{{drawing:d08_data_flywheel}}

## 1. Record

[Record](../learn/data/record.md). In the simulator, `start_recording` on the session and `run_policy` with `n_episodes` writes one LeRobot v3 episode per rollout; on hardware, `lerobot_teleoperate` runs a managed `lerobot-record` session from a leader arm. Fifty episodes of one task is the usual first dataset. What you name the cameras here is the embodiment the checkpoint will expect later; the `mock` policy is fine for testing the pipe and useless as training data, and its report says so.

## 2. Verify

[Verify](../learn/data/verify.md). `verify_dataset` reads the parquet back and proves the episodes are there, each with frames, each camera with its video, no dead control column. Run it before training; a trainer failing an hour in is the expensive way to learn the same thing.

## 3. Label

[Label and judge](../learn/data/label-and-judge.md). Predicates decide success per episode; a judge agent adds a grade and a failure tag; a filter keeps the episodes worth training on. The judge annotates, it never overturns.

## 4. Train

[Training](../learn/training/index.md), then [LeRobot](../learn/training/lerobot.md). `create_trainer("lerobot", ...)` with a `TrainSpec` naming the dataset, the base checkpoint and the run size; `validate()` refuses a spec it will not run before anything is built. A laptop trains a small ACT in an evening; a GPU host or the SageMaker trainer runs the same spec faster.

## 5. Deploy and measure

[Same checkpoint](first-policy.md), with your run's path in `pretrained_name_or_path`: `create_policy("lerobot_local", pretrained_name_or_path=..., embodiment="so101")` and `run_policy(policy_object=...)` on the simulated arm, then on the real one behind the gate. `eval_policy` in simulation, or labelled episodes on hardware, give the number.

You now have a policy that learned your task on your robot, and a number with its source. If the number is low, the wheel turns again from step 1 with the failures the judge tagged. Next rung: [Fleet](fleet.md).
