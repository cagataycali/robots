---
description: "The embodiment map: how one checkpoint's numbers reach a simulated body in radians and a physical body in driver units without the checkpoint knowing which."
---

# Embodiments

A learned policy is a function from a tensor to a tensor. `observation.state` goes in with a fixed width and fixed units; `action` comes out the same way. Nothing in the checkpoint says which joint is element 3 or whether the number is a radian. The embodiment map does.

{{drawing:d03_same_checkpoint}}

## What the map holds

`EmbodimentMap` in the `lerobot_local` provider is declarative: `state_keys` and `action_keys` (which robot keys fill which tensor slots, in order), `state_units` and `action_units` (`native` or `degrees`), `dim_policy` (`strict`, `pad` or `truncate` when the checkpoint's width differs from the robot's), `gripper_index` and `gripper_joint_range` (a gripper trained on 0 to 100 meets a hinge in radians), and `obs_rename` (which camera feeds which image feature). The shipped table has {{n:embodiments}} entries with aliases; `create_policy(..., embodiment="so101")` picks one by name, and an inline dict works for a robot the table does not know.

## Two dialects, one entry

The SO-101 shows why the map exists. In MuJoCo the arm reports joints `1` to `6` in radians. Through the lerobot driver the same arm reports `shoulder_pan.pos` to `gripper.pos`, in degrees with the gripper on 0 to 100, because that is how the checkpoint was trained. `embodiment="so101"` declares the sim keys and the degree units, so in sim it converts radians to degrees on the way in and back on the way out; when the observation carries none of the declared keys but does carry `.pos` keys, it binds those in motor order and converts nothing. The same name works on both bodies, and the checkpoint is never told which one it is driving. [Same checkpoint](../start/first-policy.md) runs this end to end.

## Cameras

Cameras travel by name. A checkpoint declares its image features (`observation.images.wrist`, say); your robot has cameras called whatever you called them. `obs_rename` in the map, or `obs_rename_override` on `create_policy`, routes one onto the other and can drop a camera the checkpoint does not read (`{"default": None}`). A camera the checkpoint needs and cannot find is refused before any download, naming the override to pass.

## When the widths differ

A checkpoint trained on a 7-joint arm meets a 6-joint body. `dim_policy="strict"` refuses; `"pad"` zero-fills the missing slot in place so the following joints keep their model index; `"truncate"` drops the extra output. A missing key is never silently skipped, because collapsing the vector would shift every later joint onto the wrong index and the arm would move wrong quietly.

## What the map does not do

It does not make a checkpoint work on a body it was not trained for. It makes the numbers reach the right joints in the right units, so that when the motion is wrong the reason is the policy, not the plumbing. Success on a new body comes from [recording and training on that body](data-flywheel.md).
