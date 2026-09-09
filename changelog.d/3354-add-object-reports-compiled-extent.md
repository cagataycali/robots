### Bug Fixes

- **simulation**: `add_object`'s success text reports the extent the geom **compiled to**
  rather than echoing the request, for every shape and not only for `mesh`. The mesh row was
  already read back off the compiled geom, on the ground that a shape consuming no `size`
  component cannot have its extent described by the request; primitives kept echoing `size`
  because "there it *is* the extent, and the geom compiles to it". That holds for `box` and
  `ellipsoid` and for no other shape. Measured on a live scene: a `sphere` given
  `size=[0.05, 0.09, 0.2]` compiles to a ball 0.05 m across in every axis and reported y and
  z extents of 0.09 m and 0.2 m; a `cylinder` given `[0.05, 0.1, 0.9]` compiles to
  `[0.05, 0.05, 0.9]` and reported a y extent its circular cross-section cannot have; a
  `capsule` asked for a 0.9 m height compiles to 0.95 m, because its two hemispherical caps
  add the radius at each end, so the request *under*-reported it; and a `plane` reported both
  a mirrored width it was never given and a third component the builder replaces with its own
  grid spacing. A one-component `sphere` also reported `size=[0.06]`, leaving the two axes it
  is equally wide in unstated. Only the report changes - the compiled geometry is identical,
  and a `box` or `ellipsoid` reports the same numbers as before, exactly (halving a float to a
  half-extent and doubling it back is exact, and the read-back resolution is a micron so it
  cannot quantise a request). A plane is reported from its two visual half-widths, since it is
  infinite for collision and MuJoCo's own bounding box for one is a ~2e10 m sentinel. The
  per-shape `size` table now also records the capsule's cap contribution, which the old
  "full height" wording contradicted.
