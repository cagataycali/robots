### Fixed: a short `target_orientation` is refused rather than zeroing the axes it does not supply

`WBCPolicy._resolve_command` writes the per-call `target_orientation` into a
zero-initialised command block with `command[4 : 4 + n_rpy] = rpy[:n_rpy]`, where
`n_rpy = min(c - 4, rpy.shape[0])`. `_validate_orientation` checked every
component's value and not the count, so a sequence carrying fewer than three
components left the axes it did not supply at `0.0` - not at the `rpy_cmd` value
that applies when the kwarg is omitted entirely.

With `rpy_cmd=[0.7, 0.8, 0.9]`, omitting the kwarg commanded `[0.7, 0.8, 0.9]`
while `target_orientation=[0.5]` commanded `[0.5, 0.0, 0.0]` and
`target_orientation=[]` commanded `[0.0, 0.0, 0.0]` - silently commanding zero
for axes the caller never mentioned, discarding orientation targets they *did*
configure, under a `success` result. The scalar spellings (`0.5`,
`np.float64(0.5)`) took the identical path, since `np.asarray(0.5).ravel()` is a
well-formed one-element array. Every component in each of those inputs is a
usable finite number, so the per-component value rule could not see any of it.

`target_velocity` - the sibling vector component of the same block, written by
the same method - already refused exactly this class of caller mistake
(`target_velocity must have at least 3 elements [vx, vy, omega], got 2`), and the
kwargs table in `docs/policies/wbc.md` already documented that arity rule for
`target_velocity` and not for `target_orientation`. `_validate_orientation` now
applies the same rule in the same order the sibling uses - coercion, then arity,
then per-component values - so both vector components of the block give one
answer to a partial triple, and the refusal names the parameter, the axes and the
count supplied. `WBCGaitPolicy._resolve_command`, whose `freq_cmd` slot pushes
rpy to `[5:8]`, calls the same inherited validator and is covered too.

A *longer* orientation is deliberately still accepted and truncated to the slots
available: every component the block has room for is honored and only the surplus
is dropped, which is what the sibling does with a packed velocity (`vel_full[:3]`).
