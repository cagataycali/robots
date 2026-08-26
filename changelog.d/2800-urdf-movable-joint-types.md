### Fixed: the Isaac backend's two URDF readers agree on what a joint type is

Two functions in `strands_robots.simulation.isaac` read a joint's `type` out of
a URDF. `joint_names.urdf_joint_names` keeps the ones that produce a named
articulation DOF and hands that list to `demangle_usd_joint_names` as the pool
of URDF names a mangled DOF name may decode to. `loaders.load_urdf` maps the
type onto a `JointDef` kind and *refuses* a type it does not recognise, naming
the set it accepts in the refusal - which makes it this package's record of
what URDF declares.

The movable set carried `spherical`. URDF declares six joint types -
`revolute`, `continuous`, `prismatic`, `fixed`, `floating`, `planar` - and that
is not one of them. So the two readers, handed the same file, disagreed:

```
type         urdf_joint_names        load_urdf
revolute     ['j1']                  j1 revolute
continuous   ['j1']                  j1 revolute
prismatic    ['j1']                  j1 prismatic
fixed        []                      j1 fixed
floating     []                      j1 fixed
planar       []                      j1 fixed
spherical    ['j1']                  ValueError: unknown joint type
                                     (expected one of ['continuous', 'fixed',
                                     'floating', 'planar', 'prismatic',
                                     'revolute'])
```

A name in the decode pool that no URDF can declare has two ways to do harm,
both reachable from a file. A `spherical` joint named `1` gave the DOF name
`tn__1_` a candidate to decode to, so the public vocabulary reported `1` - the
name of a joint the format cannot express. And `a-b` and `a.b` both substitute
to `a_b` under the legacy mangle, so a `revolute` joint named `a-b` beside a
`spherical` one named `a.b` made that decode ambiguous; an ambiguous decode
keeps the USD name, so the joint the URDF really declares surfaced as `a_b` on
`robot_joint_names`, on `get_observation` keys and in `send_action` resolution.
That is the #1900 leak the module exists to close, reintroduced by a candidate
URDF cannot declare.

No valid URDF reaches either, because no valid URDF carries the type: none of
the 68 URDFs the registry downloads declares one, and `load_urdf` refuses such
a file outright. The fix is the vocabulary and the documented accepted set,
which said `spherical` produces a DOF.

The set is now pinned between the loader's two answers - every member must be a
type the loader recognises, and no type the loader reads as moving may be left
out. The bounds are deliberately not collapsed into an equality: were the
importer found to surface a named DOF for a type the loader reads as `fixed`,
that belongs in the movable set without the loader's own mapping changing.

The comment's reason for excluding `floating` and `planar` also said the
registry URDFs do not use them. Five of the 68 declare a `floating` base - a1,
aliengo, go1, laikago and b2 - so the exclusion rests on the importer not
mapping a multi-DOF joint onto single named DOFs, which is why the loader reads
both as `fixed` too, and not on their absence.
