### Fixed: every simulation backend validates an `add_object` colour

`coerce_rgba` is now the single definition of the colour contract - three
components read as RGB and completed with an opaque alpha, four read as RGBA
verbatim, any other count refused - and lives in `strands_robots.utils` beside
the pose domain rather than in a MuJoCo-private module. The MuJoCo backend has
honored those counts on `add_object` and `set_geom_properties`; the Newton and
Isaac backends read the caller's `color` directly and diverged from that domain
in both directions.

Newton tested the vector for truthiness (`color or [0.5, 0.5, 0.5, 1.0]`), so a
NumPy colour - what a palette lookup or any colour arithmetic produces, and what
the `Args` advertise - raised a bare `ValueError: truth value of an array with
more than one element is ambiguous` straight through the structured envelope the
method documents as its only failure channel, while an empty vector read as
*omitted* and the default grey was painted under a success result. Everything
else was stored verbatim: a 1- or 2-component colour, a `nan`/`inf` channel, a
`bool` read as the channel `1.0`, and a bare string stored AS the colour.
`_add_object_to_builder` then handed `tuple(obj.color[:3])` to the solver at
rebuild time, reported nowhere near the call that supplied it. Isaac forwarded
the colour raw and then truncated it - `_construct_shape_prim` writes
`list(color)[:3]`, so a 5-component request was applied as its first 3 under a
success result and `"abcd"` was split per character into the colour
`['a', 'b', 'c']`.

Both backends now refuse those values with the same message MuJoCo already
emits, and an accepted colour reaches every backend as exactly 4 plain floats,
which makes the `color[:3]` reads the shape builders do well-defined by
construction rather than by the caller's discipline. A `bool` channel is now
refused for a stated reason on every shared-vector surface instead of as a
generic non-number. MuJoCo's own colour behaviour is unchanged: all 30
accepted-value, refused-value and message rows are byte-identical.

`size` and `mass` on those two backends keep their current contract - `size`
counts are shape-dependent and documented differently per backend, and `mass=0`
is documented on Newton as "make it static" - and stay pinned as out of scope.
