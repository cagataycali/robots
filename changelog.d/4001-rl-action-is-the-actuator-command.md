### Docs: the RL action is the actuator command, so `action_scale` bounds what a policy can reach

`SimEnv.step` sends `action * action_scale` to `send_action`, which is an
actuator command - a position target on a position-actuated robot, a torque on a
torque-actuated one. `action_scale` was documented as "the magnitude bound on how
far one action step may move a joint", a per-step displacement limit it is not:
on a real so100, a constant `Elbow` command of `1.0` held for 200 steps moves at
most `0.0085` rad per step at `action_scale=0.1` and converges on `0.1084` rad,
where it stays for any number of steps - so a target at `0.9` rad, well inside
the `Elbow`'s `[-0.174, 3.14]` ctrlrange, is unreachable at that scale.

The consequence is now stated where the contract is: the backend clamps the
product to each actuator's `ctrlrange`, so an actor whose output is bounded (the
`tanh`-squashed FastSAC / FastTD3 actors emit `[-1, 1]`) reaches only the overlap
of that range with `[-action_scale, action_scale]` - 46.4% of the so100's six
ranges at the default scale, 57.6% of the g1's twenty-nine and 3.5% of the go2's
twelve torque limits. Both RL training examples' "swapping the robot and the
reward terms" recipe names the scale too, and `n_substeps` no longer calls the
action a position target unconditionally.
