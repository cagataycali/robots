### Fixed: a URDF link's pose is read from the joint that reaches it, instead of being discarded

`load_urdf` gave every link `position=(0.0, 0.0, 0.0)` and left `BodyDef.orientation` at its
identity default, discarding the `<joint><origin>` that is the only place URDF states where a
link sits. The comment justifying the discard said the absolute pose would be "computed by
joint chain at instantiation time", but `JointDef` carries no origin, so the offsets were held
nowhere in the returned `ProceduralRobot`: a seven-link arm and a seven-link pile at the origin
were the same report, and the load reported success either way. Across the shipped asset corpus
572 of 601 joints in all 27 URDFs declare a non-zero `<origin>`, so every URDF the registry
ships was affected.

The sibling reader in the same module, `load_mjcf`, reports both halves of a link's pose in its
parent's frame. Each link's pose now comes from the `<origin>` of the joint that reaches it --
`xyz` into `position` and `rpy` into `orientation` -- because URDF places a link on that joint
rather than on the `<link>` element. A root link, reached by no joint, keeps the identity pose,
which is its placement in the model frame. `rpy` needs none of the lookups the MJCF reader does:
URDF has no `<compiler angle>`, so the triple is always radians, and it always names rotations
about the fixed axes applied roll then pitch then yaw, so the composed rotation is `Rz(yaw) *
Ry(pitch) * Rx(roll)`. An absent or malformed triple reads as identity, matching `_parse_xyz`'s
tolerant reading of the sibling `xyz` attribute on the same element.

The origins are applied after `_validate_kinematic_tree`. That guard establishes that at most
one joint reaches each body, which is what makes "the joint that reaches this link" a single
origin rather than a choice between two, so a link with two parents is still refused rather than
having one of its two origins picked.

Graded against `mujoco.MjModel`, which parses URDF and stores `body_pos` / `body_quat` in the
same parent-relative frame `BodyDef.position` reports: agreement rises from 29 of 127 links to
127 of 127 across every shipped URDF the compiler can read, with no link moving away and no
change to which files load. 98 of those are direct parent-for-parent comparisons; the other 29
are bodies the compiler re-parents -- it merges a URDF's root link into `world` and welds a
`fixed`-joint link into its parent, while the loader keeps every `<link>` as its own body -- and
they reconcile exactly by composing the loader's chain, which the new suite pins.
