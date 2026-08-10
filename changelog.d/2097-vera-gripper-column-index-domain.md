### Fixed: `decode_vera_delta_chunk_to_targets` holds `gripper_dim_index` to the columns a chunk has

The gripper column was selected with `gidx = gripper_dim_index if gripper_dim_index >= 0 else D - 1`,
a test that doubles as the only check: whatever failed it was read as the `-1` "trailing column"
sentinel. So `-5`, `-99` and `nan` were each answered with the *default* trailing column - the value
the caller meant and the value used differed, with nothing logged and nothing in the result to say so -
while `2.7`, `inf`, `True`, a numeric string, `None`, a list and an index past the last column reached
`action_chunk[:, gidx]` or `np.delete` and raised `IndexError`/`TypeError` from numpy about an axis,
naming neither the parameter nor the function and missing the `ValueError` channel the function
documents. The value is not only a caller's: the provider reads it from the policy server's metadata
and forwards it here.

It now routes through `coerce_gripper_dim_index`, beside the sibling that owns the rotation encoding
width, which delegates numeric-ness, `bool` rejection, finiteness and the float64 range to
`finite_number_error` and decides only the sentinel and the sign; an in-range index that names no
column of the chunk is reported against that chunk's width. As for `rotation_dim`, the accepted value
is normalized to an `int`, so an integral float - what a config read produces, and what
`int(meta["gripper_dim_index"])` produced on the provider path - now decodes instead of failing to
index. The index is read only when `has_gripper`, so it is checked only then.
