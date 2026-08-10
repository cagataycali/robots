### Fixed: a refused weld-removal recompile no longer drops the constraint from the live scene spec

`remove_equality_constraint` deleted the named equality from the live `MjSpec`
and then recompiled, with no way back if the recompile was refused. `detach_bodies`
reported the removal as failed while the constraint was already gone from the spec
and still present in the compiled model, so the identical retry was refused
permanently as "not found" and the next unrelated scene mutation -- an `add_object`
that never mentions the weld -- recompiled the spec and silently released the pair.

It now snapshots the spec before the delete and restores it when the recompile is
refused, matching the way its inverse `add_weld_constraint` already deletes the
equality it had just added. A refused removal is a no-op, so the attachment
`attach_bodies` recorded stays true and the identical `detach_bodies` succeeds once
the refusal clears.
