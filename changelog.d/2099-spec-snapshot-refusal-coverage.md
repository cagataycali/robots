### Quality: pin the spec-snapshot refusal contract every scene mutation depends on

Every MuJoCo scene mutation that is only validated by the recompile it precedes
takes a deep copy of the live `MjSpec` first, and that copy is the only way
back. `_snapshot_spec`'s docstring states the consequence for a caller -- "A
caller that cannot snapshot must refuse its mutation rather than proceed with no
way back" -- because a mutation applied with nothing to restore leaves an orphan
in the spec, and an orphan makes every later scene mutation fail to recompile.

That refusal had never been exercised. The helper's failure branch and all three
callers' refusals were unreached, so nothing verified that `add_robot`,
`actuate_robot` and `patch_scene_mjcf` decline rather than mutate blind -- each
through a different channel. The new tests drive a refused `MjSpec.copy` through
all three public surfaces and assert the whole cost of the refusal: the compiled
model is unchanged, the mutation's own element is absent, the scene is still
mutable, and the identical call succeeds once the failure clears.
