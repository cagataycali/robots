---
description: "Record what a robot did, verify and label it, train a checkpoint from it, run the checkpoint back on the robot, and measure: the loop that turns a demo into a policy."
---

# Data flywheel

A policy that works on your robot was trained on your robot. The flywheel is the loop that gets you there, and every turn of it goes through the same `Robot` object and the same `run_policy` call the rest of the site uses.

{{drawing:d08_data_flywheel}}

## Record

`DatasetRecorder` writes LeRobot-format episodes: one frame per control step with the observation the policy would see (joint state, camera images) and the action that was applied. In simulation the recorder is a tool action pair, `start_recording` and `stop_recording`, or the `video=` argument of `run_policy`; on hardware it wraps a teleoperation session or a rollout. What it stores is what a policy will later be trained on, so the camera names and units at record time are the embodiment the checkpoint will expect. [Record](../learn/data/record.md) walks through one episode.

## Verify and label

`verify_dataset` reads the parquet back and proves, from the files and not from anyone's narration, that the dataset holds the episodes you intended, with frames in every one, video files that match and no dead control column. [Label and judge](../learn/data/label-and-judge.md) adds a per-episode verdict the simulator's own predicates decided, and a quality grade and failure tag a judge agent writes on top; the judge annotates a verdict, it never overturns one.

## Train

`create_trainer(provider, **spec)` builds a `Trainer`; `TrainSpec` names the dataset, the base checkpoint, the output directory and the run size, and `validate()` refuses a spec before anything is built: a path that escapes its directory, a value a backend would read as a flag, an `extra` key that would set an arbitrary config attribute, a run size that is not a positive integer. The lerobot trainer runs on the machine you are on; the SageMaker trainer runs the same spec on AWS; the RL trainers run in batched simulation. [Training](../learn/training/index.md) lists what trains where.

## Deploy and measure

The checkpoint that comes out is a `pretrained_name_or_path` like any other: `create_policy("lerobot_local", pretrained_name_or_path=<your run>, embodiment=<your robot>)` and `run_policy(policy_object=...)` on the simulator first, then on the real arm behind the gate. `eval_policy` and the benchmark predicates give a success rate in simulation; on hardware the label step gives it. A number that comes out of this step, with the dataset and the checkpoint that produced it, is what earns a row on a [robot page](../robots/index.md).

## Stream and sync

Datasets outgrow laptops. [Stream and sync](../learn/data/stream-and-sync.md) puts a dataset on the Hub or in an HF Storage Bucket and reads frames back from either without downloading the whole thing.

## Where the loop closes

Teach it, the Stage 4 rung of the [ladder](../start/index.md), is one turn of this wheel on an SO-101: record, verify, train, run the checkpoint on the same arm. The [same checkpoint](../start/first-policy.md) page is what the last step looks like when it works.
