### Fixed: a tab-separated OBJ asset parses as the space-separated one does

`_parse_obj` recognised an OBJ keyword by matching a `"v "` / `"f "` prefix.
OBJ is whitespace-delimited, so a keyword may be followed by a tab as
legitimately as by a space, and MuJoCo - the reference reader for every
asset this module is pointed at - reads both forms identically. Matching the
prefix skipped a tab-separated vertex line, and the resulting failure never
named the vertex: an all-tab tetrahedron was refused as `has no triangle
geometry (empty vertices/faces)` despite declaring four vertices and four
faces, tab vertices beside space faces were reported as `face index 1 out of
range (0 vertices declared)` against the face on line 5, and one writer
emitting both separators silently dropped half the vertices so that every
later 1-based face reference landed on the wrong one.

The quiet case is the one that reaches a scene. A tab-separated vertex that
no face references was dropped with nothing raised, so `mesh_aabb` reported
bounds the file never declared - a 9-unit-tall asset came back with a zero z
extent - and `mesh_aabb` is what the MJCF scene loader turns into a
mesh-only body's collision proxy. That is the eval-integrity failure mode
the default box proxy was removed for, arriving under `status="success"`.

The keyword is now read off `line.split()`. `vn` / `vt` fall out naturally,
being different keywords rather than the vertex keyword followed by a
separator, and the ASCII-STL parser in the same module had always read its
keyword this way - so one module no longer answers "is a tab a separator"
two different ways. Every refusal keeps its exact wording and line number,
and all 277 `.obj` / `.stl` / `.msh` assets in the LIBERO tree parse
byte-identically: same vertex and face counts, same geometry, same
`mesh_aabb` centre and extent.
