### Quality: pin what a rejected `patch_scene_mjcf` batch leaves behind

`tests/simulation/mujoco/test_scene_patch_rollback_restore_failure.py` stubbed
`SpecBuilder.from_mjcf_string` to force a "the rollback's own restore failed"
branch. That branch does not exist: since the spec snapshot moved to
`MjSpec.copy`, `patch_scene_mjcf`'s rollback is a plain reassignment of the
cached spec, and `from_mjcf_string` is reachable only from
`replace_scene_mjcf`. The stub intercepted nothing, so both tests passed on the
ordinary atomic-rollback path while the module docstring told readers the
snapshot is taken via `spec.to_xml()`.

Renamed to `test_patch_scene_mjcf_rejected_batch_recovery.py` and re-pointed at
what a caller can observe after a rejection: the message names which op failed,
the world still steps, and -- previously unpinned anywhere -- the **live spec**
is the clean snapshot, so a later mutation recompiles without the rejected
batch's body. Deleting the restore, or restoring the mutated spec instead of the
snapshot, was invisible to the whole existing suite.
