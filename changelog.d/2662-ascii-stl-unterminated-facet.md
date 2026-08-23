### Fixed: an ASCII-STL facet with no `endfacet` is refused instead of dropped

`_parse_stl_ascii` flushed a facet when it read an `endfacet` line and kept no
state about one still being open. A file whose last facet was unterminated
therefore parsed successfully, one triangle short, with that triangle's three
vertices still in the returned points - and so in the bounds `mesh_aabb`
reports and the `extent` `convert_mesh_to_usd` authors. Nothing was raised.
The MJCF scene loader turns those bounds into a mesh-only body's collision
proxy, so such an asset got a proxy spanning geometry its own mesh does not
carry: the eval-integrity failure mode the default box proxy was removed for,
arriving under `status="success"`.

Two other spellings of the same missing state were loud but misdiagnosed. A
file whose only facet was unterminated was refused as `has no triangle
geometry (empty vertices/faces)`, for a file that declares a facet with three
vertices. A `facet` keyword arriving before the previous `endfacet`
accumulated both facets' vertices into one, so the file was refused for
`facet with 6 vertices (expected 3)` - an arity no facet in it declares -
against the line whose `endfacet` was the only correct thing about it.

The open facet is now tracked by the line its first vertex is on, and refused
at both ends, the following `facet` keyword and the end of the file, through
one shared wording. The binary STL and legacy MSH parsers in the same module
already refuse a truncated file by reconciling their declared counts against
the byte length; an ASCII STL declares no counts, so the open facet is the
only thing there is to reconcile. `facet with N vertices (expected 3)` is
unchanged for a facet that really does declare the wrong number between its
own `facet` and `endfacet` lines, and no other parser is touched.
