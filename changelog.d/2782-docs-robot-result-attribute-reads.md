### Fixed: the mesh teleop example builds the teleoperator it publishes from

`docs/mesh.md` opened its mesh-teleop recipe with
`leader.start_teleop_publish(teleoperator=leader.teleoperator, ...)`. There is no `teleoperator`
attribute on what `Robot("so100", mode="real", mesh=True)` returns: it is on no class in the
wrapper's MRO, and no method assigns `self.teleoperator`. The documented bring-up line raised
`AttributeError` on the first statement a reader ran. `start_teleop_publish` wants an object with
`get_action()`, and the documented way to build one is the `Teleoperator()` factory, so the example
now builds it the way `docs/hardware/teleoperation.md` does.

The class of error is worth guarding rather than only correcting, because `Robot()` is polymorphic
and the risky spelling is indistinguishable from the correct one. `mode="sim"` returns a
`Simulation`; `mode="real"` returns the `hardware_robot.Robot` wrapper *unless* the registry entry
declares `hardware.driver = "strands"`, in which case the factory returns the native driver itself,
whose surface is `DRIVER_SURFACE`. So `robot.attach_teleop(...)` is a correct line for a
lerobot-backed robot and an `AttributeError` for a natively-driven one, written identically.

Neither existing grader over that documentation can see it. `test_docs_real_mode_invocations`
grades the robot name and the keywords *inside* the `Robot(...)` call;
`test_docs_python_examples_are_callable` grades keyword sets against signatures, and its
`_accepted_keywords` reports "any keyword binds" for a callee carrying `**kwargs`, which `Robot`
does. Attribute access on the factory's return value is outside both, which is why both stay green
over the line above -- before and after it is corrected.

`test_docs_robot_attribute_reads_resolve` closes that gap: it resolves each documented read to the
type the factory would return for that `(name, mode)` pair and requires the name to exist there.
The surface is the union of three sources, because an attribute can arrive from any of them -- the
class and its MRO, `self.X = ...` assigned anywhere in the MRO, and `instance.X = ...` bound by the
factory in `strands_robots.robot`. That third source is load-bearing rather than defensive: `run`,
`mesh` and `peer_id` are bound there and appear on no class, so a surface derived from the classes
alone reports the documented `Robot("so100").run()` as missing.

Across 58 documentation pages the scan finds 65 distinct reads, 5 pages contributing hardware-mode
reads and 20 contributing simulation-mode ones. One was unresolvable, and it is the line above; the
remaining 64 resolve unchanged, including every `mode="real"` read for a robot whose registry entry
declares no native driver. Because the corpus is expected to be clean, the rule is graded over
constructed exemplars as well, so it cannot pass by having nothing left to classify.
