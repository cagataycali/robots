### Fixed: a URDF joint's axis default comes from URDF, not from MJCF

`_parse_axis` is the shared 3-vector parser behind both XML loaders in
`strands_robots/simulation/isaac/loaders.py`, and it carried a single default in its own
signature, `(0.0, 0.0, 1.0)`. That is MJCF's default: an MJCF `<joint>` that omits `axis` acts
about +Z. URDF's default is +X, and `<axis>` is optional there, so `load_urdf` -- which called the
parser without naming a default -- reported a URDF joint that omits `<axis>` as acting about +Z.
Both are valid axes and `JointDef.axis` is the whole product of the read, so nothing could tell a
joint the file declares in one plane from one whose axis had been replaced by the other format's,
and the load reported success either way. The borrowed axis was pinned by a shipped test whose
name stated it, `test_axis_wrong_arity_and_non_numeric_fall_back_to_z`, even though that test's
own docstring says malformed input must fall back to "the documented default".

Each format now names its own default -- `_URDF_DEFAULT_JOINT_AXIS` beside the URDF joint-type
map, `_MJCF_DEFAULT_JOINT_AXIS` beside the MJCF one -- and `_parse_axis`'s `default` is required
rather than carrying a value of its own. A signature default is exactly what let one format's
axis reach the other format's reader, so a call site added later cannot repeat this by saying
nothing: the omission is a `TypeError`, not a wrong axis. The third call site in the module, an
MJCF mesh scale, already passed its own default, so the two joint readers now follow a pattern
the file had already established.

MJCF's reading is unchanged, and the same oracle confirms it was already right: MuJoCo parses
URDF as well as MJCF, so `jnt_axis` answers the same question for both formats, and it reads an
absent axis as +X for `revolute`, `continuous` and `prismatic` alike and as +Z for `hinge` and
`slide`. That is what places the correction at the URDF call site rather than in the shared
parser. Every expectation in the new suite is derived from the compiler; no expected axis is
restated by hand.

The parser's tolerance is deliberately left alone: an `<axis>` stating a vector it cannot read
still degrades to the format's default rather than raising, even though MuJoCo refuses every such
model outright. Only which default the tolerance lands on moves here, and the boundary is pinned
either way so that moving it later is an explicit decision.

Across the downloaded asset corpus -- 74 URDFs, 2042 joints, with identical load and refuse
verdicts on both readings -- no movable joint changes: all 1196 `revolute` / `continuous` /
`prismatic` joints declare their own `<axis>`, so the corpus never exercised the default and the
defect was latent. The 486 joints that do change declare no axis at all (483 `fixed`, 3
`floating`), every one of them is reported with `joint_type="fixed"`, and MuJoCo produces no hinge
or slide for them, so their axis is not a quantity anything acts on.
