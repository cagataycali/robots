### Fixed: `simulation/isaac` - a URDF joint that states no type is refused, not welded

`type` is a required attribute of a URDF `<joint>`, so an absent one is missing
information rather than a declaration of a default. Both of this package's URDF
readers defaulted it to `"fixed"`: `load_urdf` emitted a `JointDef`
byte-identical to the one a deliberate `type="fixed"` produces, and
`urdf_joint_names` dropped the joint from the movable set. So a file that failed
to declare a joint loaded as a robot with fewer actuated DOFs than it names, the
load reporting success, and no caller able to tell it from an author welding
that joint on purpose - the silent `joint_count` the module's failure semantics
exist to convert into a message.

The empty spelling of the same omission was already refused by name (`type=""`
-> "unknown joint type"), so one file was refused and the other welded for the
same missing declaration, decided by whether the attribute was written empty or
left out. MuJoCo, which this loader already cites as its reference for the URDF
`<axis>` default, refuses both spellings: `required attribute missing: 'type'`
and `invalid joint type in URDF joint definition`.

Both readers now refuse an absent `type` in their own voice, alongside the
`name` refusal each already raised. A joint that STATES a type outside the
movable set is still skipped by `urdf_joint_names` rather than refused - a
declared type it has no named DOF for is not a malformed file, and `load_urdf`
remains the reader that grades the vocabulary. MJCF is deliberately unchanged:
that format documents `hinge` as the default for an omitted `type`, so an absent
one there IS a declaration, and the regression test pins the contrast.
