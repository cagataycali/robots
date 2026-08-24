### Quality: `add_camera` applies the pose rule once, not twice

`MuJoCoSimEngine.add_camera` validated `position` and `target` twice - a
`coerce_pose_vector` call per parameter, then a loop re-running the non-coercing
twin `pose_vector_error` over the values that call had just returned. The second
application could not refuse anything: both helpers read a pose through the same
`_read_pose_vector`, so a value the coercing guard accepts is by construction one
the twin accepts, and the two substituted defaults are literal finite 3-vectors.
Measured on a 20-value probe set across both parameters, with the second
application neutered to always return `None`, it was invoked 24 times and 0 of 40
outcomes changed.

It was dead code carrying a live comment - the comment justified the loop with
the two failures the guard above it now owns - so a reader asking which check
owned the pose contract found two answers, and the line was permanently
unreachable by any test. The loop and its now-unused import are removed, leaving
one owner per pose parameter; the accepted and refused domains are unchanged.
Re-inserting the loop verbatim is invisible to all 2,937 pre-existing MuJoCo
backend tests, so the invariant that makes one owner sufficient is now pinned:
one application per parameter, the coercing guard's totality over the twin's
domain, and the degenerate-orientation comparison below it reading a substituted
default and a normalized NumPy pose.
