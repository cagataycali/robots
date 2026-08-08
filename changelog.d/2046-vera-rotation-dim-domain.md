### Fixed: refuse a VERA rotation-encoding width the eef-delta decoder cannot read

`rotation_dim` selects which rotation parameterization the VERA eef-delta decoder
reads out of each end-effector delta, and `delta_to_matrix` implements exactly two
of them — axis-angle (3) and rot6d (6) — raising for any other width. Neither
public surface that takes the width held it to that enumeration:
`VeraPolicy.set_ik_target` stored `int(rotation_dim)` and
`decode_vera_delta_chunk_to_targets` took it verbatim, while the sibling
`translation_scale` was already checked in both signatures.

Six widths (`0`, `-3`, `2`, `4`, `2.7`, `True`) were therefore stored and refused
only mid-rollout, from inside `get_actions` on the first inference — after the
policy-server handshake and the IK bridge build — rather than at the call that
supplied them, and by then the setter had already written the model, the
end-effector frame and reset the bridge. Because `int()` truncates before storage,
`2.7` was reported as `unsupported rotation_dim 2` and `True` as `1`: a width the
caller never supplied. `inf` and a list raised `OverflowError` / `TypeError` out of
that coercion instead of the `ValueError` the method uses for its sibling.

The two surfaces also disagreed about the same value: `3.0` and `"6"` were honored
through the setter (which coerced them) and raised `TypeError: slice indices must
be integers` in the decoder, out of the per-step rotation slice.

Both now route through one owner beside the dispatch that defines the enumeration,
which delegates numeric-ness, `bool` rejection and finiteness to the shared
`finite_number_error` domain and decides only membership. An integral float stays
accepted and is normalized to an index, since the width slices the rotation block;
a numeric string is now refused at both surfaces rather than coerced at one.
