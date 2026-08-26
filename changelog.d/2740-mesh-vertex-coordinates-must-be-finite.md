### Fixed: a mesh asset declaring a vertex coordinate that is not finite is refused, not measured around

Two things measure the vertices `load_mesh_geometry` returns: `mesh_aabb`, whose `(center, size)`
becomes a scene object's collision proxy, and the `extent` `convert_mesh_to_usd` authors into the
converted USD. Neither reported a non-finite coordinate.

`min`/`max` order a NaN as neither smaller nor larger than anything, so `mesh_aabb`'s running
comparison dropped such a vertex and returned the bounds of the vertices that *were* finite -
numerically indistinguishable from a mesh that declared only those, under no error at all. That is
exactly the outcome the module's ASCII-STL refusal for an unterminated facet already exists to
prevent, reached from the other side: there the vertices are in the bounds and their triangle is in
no face, here the vertex is in a face and not in the bounds. The `extent` is built by a different
spelling - `min` over an iterable rather than a running comparison - which keeps a leading NaN and
drops a trailing one, so one module could measure one file two ways depending on the order its
vertices happened to be declared in. An infinite coordinate needed no such subtlety: the reported
extent was simply unbounded, and a scene object sized from it is a collision proxy with no bounds.

Measured on a four-vertex `.msh` tetrahedron whose vertex 0 is extremal on -x: replacing that one
coordinate with NaN collapsed the reported proxy from `2.5 m` to `1.0 m` on that axis and shifted
its centre from `-0.25` to `0.5`, and `load_mjcf_scene_objects` returned that scene object under
`status success` for an MJCF MuJoCo refuses to compile. Rendered headless, a quarter of the asset's
silhouette (24123 pixels) falls outside its own reported proxy, against 0 for the same asset with
four finite vertices.

One finiteness rule is now applied where the four parse paths meet - OBJ, ASCII STL, binary STL and
legacy MuJoCo MSH - so no format can drift into tolerating what its siblings refuse, and so both
bounds consumers are covered by the reader they share rather than each needing its own guard. The
refusal names the offending vertex by index, which is the locator MuJoCo reports for the same
refusal and the only one that works for the two binary paths. The fast path is a single C-level
`map` over the flattened coordinates, measured at ~4% of the parse it follows across the 60 largest
shipped meshes; the per-vertex walk that finds the offender runs only after that map has already
failed. It is a finiteness rule and not a magnitude bound, so a legitimately huge or vanishing
coordinate is still a position - float32 max, its negative, float32 tiny, `-0.0` and `1e300` are
all pinned as accepted.

Refusing is the format family's own disposition rather than a local policy: MuJoCo owns `.msh` and
reads the other two, and it refuses the same input with `vertex coordinate N is not finite`. All
twelve fixture and format combinations now agree with it, and the six MuJoCo readings are pinned as
premises so the agreement is graded rather than asserted.

`convert_mesh_to_usd` also reads the asset before probing for `pxr`, the way the extension and
existence checks above it already did. An install without the `sim-isaac` extra used to be told to
install `usd-core` for an asset that was simply broken; it is now told what is wrong with the mesh
it handed over. The content-addressed cache hit still short-circuits both, and a cache entry only
exists because an earlier call parsed the same bytes successfully.

Nothing legitimate is refused: all 10469 mesh files in the downloaded robot-description corpus -
3499 STL, 6882 OBJ, 88 MSH - parse unchanged, with zero non-finite coordinates among them, so the
defect was latent rather than live in shipped assets. Deliberately unchanged: a USD input, which is
referenced verbatim and whose vertices this module never reads, so the rule is scoped to the formats
it parses.
