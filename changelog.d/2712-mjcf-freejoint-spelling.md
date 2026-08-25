### Fixed: both MJCF spellings of a free joint are read, instead of only one

MJCF states a floating base two ways and MuJoCo compiles both to the same `mjJNT_FREE`:
`<joint type="free">` and the dedicated `<freejoint>` element. `load_mjcf` walked
`body_el.findall("joint")`, so only the first spelling produced a `JointDef` and a `<freejoint>`
produced none at all - the base's six degrees of freedom were absent from the returned
`ProceduralRobot` rather than reported as a joint a caller can see and skip. The scene reader in
the same module already consulted both spellings, and the module's own docstring already named
them as equivalent, so the robot reader was the one surface that knew about only one.

`<freejoint>` is the spelling the shipped asset corpus uses. Of the 46 loadable robot MJCFs under
`robot_descriptions` that declare a free joint, every one spells it `<freejoint>` and none spells
`type="free"`, so a floating base was reported for none of them: the quadrupeds (`unitree_go2`,
`anymal_c`, `spot`), the humanoids (`unitree_h1`, `talos`, `berkeley_humanoid`), and most visibly
the aircraft - `skydio_x2` and `bitcraze_crazyflie_2` have exactly one joint each, that base, so
the loader reported a robot with no joints whatsoever.

Both tags are now read, in document order. MuJoCo fixes the two rules that makes this
unambiguous, and each is pinned as a premise. A free joint may not share a body with any other
joint (`more than 6 dofs in body`), so a body carrying a `<freejoint>` carries nothing else and
the order of the two tags can never differ - which also means the change cannot reach the
`_validate_kinematic_tree` compound-joint guard. And `<freejoint>` resolves no default class:
MJCF has no `<default><freejoint>` block, and MuJoCo gives such a joint the built-in damping and
armature even where a `<default><joint>` class is in force, while the `type="free"` spelling does
inherit that class. The loader now diverges in exactly that place and in exactly that direction.

Graded across the corpus by rewriting every `<freejoint .../>` as `<joint ... type="free"/>` and
asking MuJoCo whether that is the same model - it is, for 48 of 48 files, with no premise
violated - the two spellings agreed on 0 of 43 gradable files before and 43 of 43 now, with 43
joints under-reported before and 0 now. 20 of the 43 reports are byte-identical field for field;
the other 23 differ only in fields a `<default><joint>` class supplies, and MuJoCo diverges the
same way on all 23. Nothing else changes: the load and refuse verdicts are the same on both
readings.

What `free` MAPS to is deliberately unchanged and pinned. `_MJCF_JOINT_TYPE_MAP` reports it as
`"fixed"` because `JointDef` has no 6-DOF spelling, so a floating base is now visible in `joints`
without being counted as an actuated DOF by `num_joints`. Whether the dataclass should gain such
a spelling is a contract question about every producer of a `ProceduralRobot`, so moving it later
has to be an explicit decision rather than a side effect of this one.
