### Fixed: two registry entries declare the joint count their own asset has

Every robot in `registry/robots.json` carries a `joints` figure, and two
discovery surfaces report it verbatim: `get_robot` returns it, and `list_robots`
prints it in the `Joints` column an agent reads to size an action vector. Two
entries disagreed with a sibling built from an indistinguishable model - same
joint names, same joint types, same actuator names - so one compiled shape was
described by two different numbers:

```
robot        declared   asset njnt   movable   actuators   sibling declares
ur5e         8          6            6         6           ur10e -> 6
unitree_a1   16         13           12        12          aliengo, go1 -> 13
```

`ur5e` is a six-axis arm with no gripper and no floating base; its own
description says so ("6-DOF industrial"), and `ur10e` - whose compiled model is
byte-identical in joint names, joint types and actuator names - declared `6`.
`unitree_a1` shares its model with `aliengo` and `go1`, which both declared `13`,
the model's `njnt` of twelve movable joints plus the floating base;
`anymal_b` and `anymal_c`, a separate quadruped family whose model has the same
13/12/12 shape, declare `13` too. Both figures are now what the shared asset has,
and `docs/robots/arms.md` and `docs/robots/mobile.md` - which rendered `8` and
`16` in tables directly above and below the siblings' correct rows - agree.

What `joints` counts registry-wide is deliberately not settled here.
`docs/robots/arms.md` says "Joint counts include any free joints / gripper
actuators", which reads as MuJoCo's `njnt` and holds for `anymal_b`/`anymal_c`
(13 against a 12-DOF description, the extra one being the base). But `panda`
declares `7` against an `njnt` of 9 - the arm without its two finger joints - and
`arx_l5`/`piper` both declare `11` against an `njnt` of 8. Of the 50 registry
robots whose asset loads, 22 declare a figure that is neither their `njnt` nor
their movable-joint count. Choosing one convention would rewrite those 22 numbers
on a guess about what each was counting, so the regression test grades a weaker
property that needs no such decision: two robots whose compiled models are
indistinguishable must be described by the same number, whatever that number is
counting. That holds under every convention above, because the models agree on
all of them - and it is the in-family control that makes these two figures
decidable while the registry-wide question stays open.

The assets group the 59 resolvable entries into six such families. Four already
agreed; the two that did not are corrected here, except `vx300s`/`wx250s`, which
declare `19` and `16` against an `njnt` of 8 while describing the same shape as
each other ("6-DOF + gripper"). Unlike the two fixed above, nothing there says
which figure is right, or whether either is - a two-member family has no majority
- so the pair is recorded as an unresolved family with that reason rather than
guessed at, and removing it from that set is how the convention decision gets
enforced. The test re-derives the grouping from the compiled models and fails if
a family it names no longer groups together, or if the assets group robots that
no family names, so the frozen table cannot drift into agreeing with a wrong
registry.
