### Bug Fixes

- **tools/pose_tool**: a pose-library write now stores the whole library or
  leaves it exactly as it was, and a write that could not happen is reported
  instead of being logged. `store_pose` and `delete_pose` are read-modify-writes
  of the *whole* `<robot_id>_poses.json` document, and the write encoded
  straight into the destination it had already truncated - so a failure partway
  through did not lose the pose being changed, it lost every posture the arm
  had. `_load_poses` reports an unparseable file as *no poses*, so the loss came
  back as an arm with nothing stored: measured on a full disk, a library holding
  four postures became 40 unparseable bytes, the next `list_poses` answered "No
  poses stored for robot hw_arm", and the `store_pose` that destroyed it
  returned `status=success` with "Stored pose 'inspect'" - one line below the
  refusal that declines to persist a partially-read arm because "a stored pose
  is a named posture every later `load_pose` drives towards". A value JSON
  cannot represent (a NumPy joint angle) did the same thing without any I/O
  failure at all. The document is now serialized in full before the destination
  is touched and committed through a temp sibling plus `os.replace` - the
  sequence `registry.user_registry._save_user_registry` documents - so an
  unencodable pose raises a `ValueError` naming the library, an I/O error during
  the commit leaves the previous library intact with no temp file behind, and
  both surface through `pose_tool` as `status=error` naming the pose that was
  not stored and the postures that are unchanged. The in-memory library is
  rolled back with it, so it never holds a pose no reader can find.
