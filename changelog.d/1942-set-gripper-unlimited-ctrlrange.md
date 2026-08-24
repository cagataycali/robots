### Fixed: an unlimited gripper actuator is driveable, not undriveable

`set_gripper` derived its open/close set-points from `model.actuator_ctrlrange`
alone and refused any actuator whose range was not strictly increasing. MuJoCo
reports exactly `(0, 0)` with `actuator_ctrllimited == 0` for an actuator the
MJCF left *unlimited*, which is a different claim from "this actuator accepts
nothing" - so the primitive refused on **so101**, a shipped robot whose registry
gripper metadata is correct and whose `move_to` and `rotate_wrist` both work:

```
set_gripper: actuator 'a/6' has no usable ctrlrange (0.0, 0.0);
cannot infer open/close set-points.
```

so101's sim MJCF declares neither `ctrlrange` nor `inheritrange="1"` on its
position servos; so100's sets `inheritrange="1"` on every actuator, which
compiles a real ctrlrange from the driven joint - the only reason so100 was
unaffected. The two jaw joints' ranges are near-identical, so nothing about
so101's gripper was undriveable; its ctrlrange simply was never authored.

For a JOINT / JOINTINPARENT-transmission position servo `ctrl` *is* the joint
target, so the driven joint's own limits are the open/close set-points - exactly
what `inheritrange="1"` would have compiled the ctrlrange to. Both sibling
primitives already made that substitution (`rotate_wrist` and `move_to` read
`jnt_range` under `jnt_limited`); `set_gripper` was the sole outlier. It now
falls back to the driven joint's range when the ctrlrange is unusable **and**
`actuator_ctrllimited == 0` **and** the actuator has a joint-transmission entry
in `_joint_actuator_map` **and** that joint is itself limited with a strictly
increasing range.

A tendon actuator keeps refusing: its ctrlrange is a normalised command space
rather than joint units (the shipped Franka gripper is `(0, 255)`), so a joint
range would command the wrong quantity. That scoping is structural rather than a
new special case - only joint transmissions appear in `_joint_actuator_map`, so
a tendon actuator has no entry to substitute from. A refusal now names every
source it exhausted instead of only the ctrlrange, and the success payload
gained `setpoint_sources`, so a substituted joint range is visible per actuator
rather than silent.

The single-point ctrlrange that looks like a deliberate restriction is not one:
MuJoCo collapses any non-strictly-increasing `ctrlrange` to
`ctrllimited == 0` and clamps `ctrl` only when `ctrllimited == 1`, so such a
range is inert and the substitution widens nothing that was being enforced. The
compiler also rejects an explicit `ctrllimited="true"` over a degenerate range
outright, which means the guard respecting such a claim is reachable only on a
model mutated after compilation - as `policies/wbc/sim_control.py` does when it
hands `ctrlrange` to a whole-body controller. Both facts are pinned as premise
tests, so a future MuJoCo that changes the encoding fails loudly and names the
reason rather than breaking the behaviour tests obscurely.
