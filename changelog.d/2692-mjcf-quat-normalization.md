### Fixed

- **isaac**: an MJCF `quat` is reported normalized, the way MuJoCo's compiler reads it. A
  quaternion describes a rotation only at unit norm, and MJCF authors routinely spell one
  non-unit - `quat="1 -1 0 0"` is the idiomatic quarter turn. `_parse_quat` reported the four
  components as written, so what the loaders handed back was not a rotation: MuJoCo's own
  `mju_quat2Mat` builds the matrix from the components as given, and for that declaration it
  yields the intended turn composed with a uniform scale of `|q|**2` - `det(R)` 8.0 instead of
  1.0, a unit axis coming back twice as long. Graded against the `body_quat` MuJoCo's compiler
  stored, 809 of 8398 robot links in the downloaded asset corpus disagreed with the compiler and
  every one of the 809 is reconciled by normalizing; the reading now agrees on 8398 of 8398, with
  no link moving away. 815 links across 140 files report a non-unit quaternion, including the
  shipped registry robots `pal_tiago_dual` (15 links), `shadow_hand`, `wonik_allegro` and
  `robotiq_2f85_v4`, plus 18 of 619 reported scene objects. `_parse_quat` is the one helper both
  loaders resolve a `quat` through, so `load_mjcf`, `load_mjcf_scene_objects` and the mesh-geom
  frame move together; the four alternative spellings (`euler`, `axisangle`, `xyaxes`, `zaxis`)
  are constructed at unit norm already and are untouched. A zero quaternion has no direction to
  normalize onto and joins the malformed readings (identity) rather than being reported as four
  zeros - MuJoCo refuses such a model outright, and it is the one orientation
  `coerce_orientation_quaternion` refuses on the write side, so reporting it was a reader handing
  out a value this library's own writers reject. Magnitude remains outside the write contract: a
  non-unit `orientation` passed into a setter is still accepted and normalized by the consumer.
  This is what a read reports, where the consumer is unknown and the compiler is the oracle.
