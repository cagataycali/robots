### Fixed: an MJCF joint declared through a `<default>` class no longer loads as a revolute joint about the wrong axis

MJCF lets a `<default class="...">` supply a `<joint>`'s `type`, `axis`,
`range`, `damping` and `armature`, so a joint element need not spell any of
them. The isaac MJCF robot loader read all five off the element, so a
class-declared joint fell to the loader's own defaults - a revolute joint
turning 360 degrees about `+z`. aloha's gripper finger is the clean case: its
element carries only a name and a class, and it was reported `revolute`,
`(0, 0, 1)`, `(-3.14159, 3.14159)` for a joint the file declares as a 41 mm
`prismatic` slide along `(0, 0, -1)` - five of five fields wrong, under
`load_mjcf` reporting success. `num_joints` and every `JointDef` are the
function's product, so a caller sizing an articulation read a translating
finger as one free to spin.

Graded against MuJoCo's own compiled model over the 1504 joints in the 97
registry assets that declare a `<joint>` inside a `<default>`, the loader now
agrees on all 1504 where it previously disagreed on 411. Every change moves
toward MuJoCo's answer and none away; 52 assets are affected, including panda,
ur5e, ur10e, iiwa14, xarm7, go2, g1, shadow_hand, allegro, leap_hand,
so_arm100, aloha and robotiq_2f85, and four grippers move from revolute to
prismatic.

The same reader already resolved a geom's class. That resolver read
`<default>/<geom>` only, so one file carried two rules for one MJCF feature;
it now takes the element tag and answers "what did this element declare" for
both `<geom>` and `<joint>`. Nested classes, `childclass`, an `<include>`d
fragment and element-over-class precedence all come from the existing rule
rather than a second implementation, and the two tag namespaces stay separate
so a class carrying both a shape and a degree of freedom cannot leak one into
the other.
