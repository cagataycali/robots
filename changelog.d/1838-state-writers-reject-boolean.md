### Fixed: the runtime state writers refuse a boolean instead of writing it as 1.0

`set_joint_positions`, `set_joint_velocities` and `apply_force` coerced their
inputs with a bare `float()`. `bool` is an `int` subclass, so `float(True)` is
`1.0` and each of them reported `status="success"` having written 1 radian,
1 rad/s or 1 N:

```
set_joint_positions({"panda/joint1": True})  -> success  "Set 1/1 joint positions, FK updated"
apply_force(body_name="panda/hand", force=[True, 0.0, 0.0]) -> success  "Force: [1.0, 0.0, 0.0] N"
```

while every scene-construction vector refused the same value
(`add_object(position=[True, 0.0, 0.3])` -> `"'position' elements must be
numbers"`). One library answered "is this a usable number" two ways for the same
kind of quantity, and the writers took the permissive answer.

They now refuse a python and a NumPy boolean, on all three surfaces and on
`apply_force`'s `force`, `torque` and `point` alike. The refusal names the joint
or vector and states the units to use instead, and because the check runs before
any write, a refused value leaves `qpos` / `qvel` and every latched wrench
untouched.

**The decision, since the code carried the opposite one.** `apply_force` held a
comment deferring this as "bool is intentionally accepted (subclass of int ->
finite); rejecting it is out of scope for numeric-element validation". That note
is retired rather than left contradicting the behaviour, and the reason now lives
next to the check. The consequence here is genuinely milder than on the actuator
command #1837 settled: there, `1.0` is re-read in each drive's own units, so the
same `True` commands a different pose on every actuator; here `1.0` is one
unambiguous quantity - 1 radian, 1 rad/s, 1 N - so a boolean is merely wrong
rather than ambiguous. The deciding argument is therefore the consistency of the
domain rather than the severity of the outcome: a caller who cannot place a body
at `True` should not be able to teleport a joint to `True` either, and no caller
can plausibly mean "1 radian" by `True`.

**One predicate, not three.** `simulation/base.py` and `mesh/security.py` each
carried their own numpy-bool unwrap, because `numpy.bool_` is not a `bool`
subclass and `isinstance(value, bool)` alone misses the boolean a comparison
(`gripper > 0.5`) produces. Rather than add a third, the predicate moved to
`strands_robots.utils.is_boolean`, beside the seven scalar domains there that
already reject a bool for the same stated reason; `send_action` now shares it.
`mesh/security.py` keeps its own inline unwrap, which raises `ValidationError`
rather than returning a structured dict and so answers to a different contract.

The gate keys on the type, not the value: `1`, `np.uint8(1)`, a 0-d numeric
array and the documented `force=[0, 0, 0]` stop are all still accepted, and
`nan` / `inf` and non-numeric values keep their own distinct messages. The
regression tests pin both directions per surface, since a gate that also
rejected `1` would have satisfied every rejection test and broken every caller.
