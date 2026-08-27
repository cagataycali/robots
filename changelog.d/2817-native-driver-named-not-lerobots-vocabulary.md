### Fixed: a robot with a native driver is named as one, not answered with lerobot's list

Two registries answer "what builds this robot": lerobot's `RobotConfig`
ChoiceRegistry and this package's native-driver registry. A robot can be in the
second and not the first - that gap is what a native driver exists to close, and
`drivers/reachy.py` records it in as many words: "the Reachy Mini has no lerobot
robot type, so before this driver `mode="real"` raised `ValueError: Unsupported
robot type: 'reachy_mini'`".

The Reachy Mini also declares `hardware.driver="strands"`, so `resolve_driver`
sends it to its driver and it never meets that refusal. Four robots the
Dynamixel driver serves - `vx300s`, `wx250s`, `trossen_wxai` and `dynamixel_2r` -
declare nothing, so the default routes them to lerobot, which has no robot type
for any of them. They reached the generic listing of lerobot's sixteen robot
types, and that listing never mentioned that this package ships the driver that
builds them. A caller with no reason to guess `driver="strands"` was at a dead
end, holding a list of sixteen names none of which was the robot they asked for.

The site already had this shape for the other wrong entry point. A leader arm is
a lerobot *teleoperator*, and `teleoperator._other_lerobot_kind_refusal` names it
as one rather than listing follower types - its docstring is where "answering it
with the names of the kind it is not answers the wrong question" is written down.
`drivers.registry._native_driver_refusal` is that function's sibling for a
natively driven robot: consulted at the same site, returning a reason or `None`
the same way, so a name with no native driver keeps the listing that is the right
answer for it. Of the fifty-five registry robots that reach this refusal, four
now name their driver and fifty-one are unchanged - including `robotiq_2f85`,
which has no driver of either kind.

It is consulted before the teleoperator arm rather than after, because the two
populations overlap: `unitree_g1` is a lerobot teleoperator type as well as a
natively driven robot, and for a name in that overlap the driver is what a
`Robot()` caller asked for, not an instruction to build the leader that
teleoperates it. Nothing observable turns on that order today - lerobot's robot
registry knows `unitree_g1` too, so it resolves and never arrives - which is why
the ordering is pinned structurally.

Resolution precedence is untouched and no registry entry gains a declaration.
Which driver wins is unchanged: `koch` and `aloha` have both a working lerobot
type and a native driver, and whether they should prefer the native one is a
preference the registry is where to declare - `unitree_g1` shows that. This
changes only what a caller is told when the driver they were routed to cannot
build the robot at all. `get_driver`'s docstring, which claimed
`hardware.driver` is "absent everywhere in the package registry, because every
robot here is driven through lerobot", is corrected alongside: two robots declare
it, and an absent declaration means "no preference" rather than "no native
driver".
