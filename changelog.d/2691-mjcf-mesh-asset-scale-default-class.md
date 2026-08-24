### Fixed: a mesh asset's `scale` is read from its MJCF `<default>` class, instead of falling back to unit scale

`_parse_mjcf_mesh_assets` read each `<asset><mesh>`'s `scale` off the element alone, but
MJCF lets a `<default>` class supply it -- `<default class="right_hand"><mesh
scale="0.001 0.001 0.001"/>` plus `<mesh class="right_hand" file="palm.obj"/>` declares a
millimetre asset with no `scale` on the element at all. Every such mesh was reported at
unit scale. That is a plausible value, so nothing downstream could tell an asset authored
at unit scale from one whose declared scale was never read: the load reported success and
the asset was reported up to a thousand times too large.

The reported scale is not cosmetic. It rides onto the visual prim's xform as
`SceneObject.mesh_scale`, and `mesh_aabb` measures the object's collision proxy `size`
through it, so a body whose only geometry is a mesh reported a proxy scaled by the same
factor -- Menagerie's `flybody` head, declared at `scale="0.1 0.1 0.1"` by the model's
root class, was reported 0.50 x 0.77 x 0.64 m instead of 0.050 x 0.077 x 0.064 m.

The reader now resolves `scale` through `_mjcf_class_defaults(root, mjcf_dir, "mesh")` and
the shared `_class_attrs` precedence, which is how the sibling `<geom>` and `<joint>`
readers in the same module already resolve their attributes: the element's own value wins,
then its named class, then the root class (reachable as `""`, as `class="main"`, or by
naming no class), with nested classes flattened and `<default>` read from the spliced
model so a class declared in an `<include>`d fragment resolves. The mesh class map is
collected separately from the geom one because one class carries a separate attribute set
per element kind. `file` and `name` stay element-only reads, which is the format's rule
rather than a simplification -- MuJoCo's schema refuses either attribute inside a
`<default><mesh>`.

Graded against `mujoco.MjSpec`'s own resolved `scale` -- the very attribute this reader
reports -- across the shipped asset corpus: agreement rises from 9864 of 10320 mesh assets
to 10320 of 10320, with no asset moving away from the compiler. The 456 corrected values
span 18 shipped Menagerie models (`flybody`, `robotis_op3`, `anybotics_anymal_b`,
`shadow_hand`, `trossen_wx250s`, `robotiq_2f85`, `stanford_tidybot`, `skydio_x2` and their
scenes), and every one of them was reported at unit scale where the model declares 0.001,
0.1 or 0.01.
