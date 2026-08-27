### Fixed: a name refusal describes the grammar that produced it, per seam

`MobileBaseRobot` validates two independently-overridable seams: a `node_name` against `_NAME_RE`,
and every topic and `init_services` service name against `_TOPIC_RE`. Both refusals appended one
shared sentence, `_NAME_HINT`.

That works only while a platform's two grammars agree, and on one platform they do not. `RtpsRobot`
writes to a DDS topic directly, so a topic must be absolute (`^/[A-Za-z0-9_/]*[A-Za-z0-9_]$`),
while its `node_name` names this robot's own agent tools rather than anything on the robot's graph
and keeps the base grammar (`^[A-Za-z0-9_/~]+$`). No single sentence is true of both seams, so the
platform's only escape was to silence the hint entirely - and the result was that the strictest
grammar in the tree was the one that said nothing about itself. `RtpsRobot("tb", "cmd_vel")` refused
with a bare `invalid cmd_vel_topic: 'cmd_vel'`, and so did `~/cmd_vel` and `/cmd_vel/`; all three are
values `RosBridgedRobot` accepts, because rclpy resolves them. The rule was already written down
twice in prose - in the comment above `_RTPS_TOPIC_RE`, and in the docstring of the test that pins
the divergence - and withheld from the caller who needed it.

The stated reason for silencing it does not survive measurement either. The comment read "`use_rtps`
requires absolute topics, so the generic hint would misdirect", but the generic hint's own example,
`/turtle1/cmd_vel`, is a topic `RtpsRobot` accepts. The sentence was not misdirecting; it was simply
attached to a constant that the other seam also had to live with.

The topic seam now carries its own `_TOPIC_HINT`, and `_check` takes the hint as an argument rather
than reading one off the class, so a seam cannot borrow another seam's sentence. `RtpsRobot` states
the absolute requirement and why a relative or `~` name cannot work there, and drops its empty
`_NAME_HINT` override so `node_name` regains the sentence that was always correct for it.
`RosBridgedRobot` overrides one grammar for both seams and so declares one sentence for both; its
messages, and the base's, are byte-for-byte unchanged, as is every grammar. The ROS 2 bridge still
accepts a relative topic.

`rtps_robot`'s module-level `_check` is removed. It had no caller anywhere in the tree, and it was a
divergent copy of the base classmethod: its refusal omitted the hint unconditionally. Leaving a
second, hint-free copy of the shared validator in the one module that had gone silent was the same
defect one edit from returning.

Why the existing suite was silent on all of it: `test_shipped_classes_keep_their_own_name_grammar`
pins the grammar divergence behaviourally and `test_rtps_robot` pins the refusal, but both match on
`"invalid cmd_vel_topic"` - the prefix, which stops immediately before the hint. Across ten
mutations of the new code, including restoring the shared constant, re-silencing either hint, and
restoring the deleted `_check`, the six pre-existing suites covering these classes report zero
failures.

The new grader derives its seam table from the bodies of the base's own `_check_*` methods and its
platform list by walking the package, so a third seam or a fourth platform is held to the rule on
arrival: each seam must forward a hint of its own, no two seams may share one, no platform's hint may
be empty, and each hint must offer an example that the pattern it describes actually accepts.
