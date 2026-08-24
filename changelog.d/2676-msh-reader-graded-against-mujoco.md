### Quality: the .msh reader is graded against MuJoCo, not against its own layout assumption

The legacy binary mesh (`.msh`) tests built every input with a helper that
writes the four header fields in the order `load_mesh_geometry` reads them and
lays the vertex, normal, texcoord and face blocks out in the order it walks
them. A round trip through that helper cannot tell a correct layout from a
mirrored one: a reader that swapped two header fields would be graded by a
fixture that swapped them too, and both halves would move together. The format
has no published specification beyond MuJoCo's own `LoadMSH`, so there was
nothing on the other side of the comparison.

The reader is now handed the same bytes as MuJoCo and required to agree with it,
over three block layouts chosen so that a swap is visible: normals and
texcoords both present, both absent, and texcoords present with normals absent -
the asymmetric case, where a reader that transposed those two header fields
computes a different total length for the same file. MuJoCo does not store the
authored frame, so the comparison recovers it as
`mesh_pos + R(mesh_quat) @ mesh_vert` and matches vertices on distance rather
than on sort order.

Two fixtures also declared three vertices, which `LoadMSH` refuses
(`nvertex < 4`), so they graded the reader against a file the format's owner
calls invalid; both now use a four-vertex tetrahedron. The refusal is pinned, as
is MuJoCo's refusal of a coplanar mesh - that second refusal is why the
comparisons need no separate extent premise.

Every `.msh` asset LIBERO ships already agrees with MuJoCo to 6.54e-06 in the
authored frame across all 88 files, so this pins correct behaviour rather than
fixing a defect; what it removes is the possibility of that agreement drifting
silently. No library behaviour changes.
