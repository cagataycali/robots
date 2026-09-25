### Fixed: reading the pose library no longer creates a directory, and an emergency stop no longer needs one

`PoseManager.__init__` ran `storage_dir.mkdir(parents=True, exist_ok=True)`, and
every `pose_tool` verb constructs a manager - including the three that only read
the library (`list_poses`, `show_pose`, `delete_pose`) and `emergency_stop`,
which is never gated because stopping is never gated. So asking what poses exist
created `.strands_robots/poses` in whatever directory the caller happened to be
in, and where that directory is read-only the same construction raised
`PermissionError` out of a call site that wraps it for `ValueError` only: an
emergency stop refused by an exception the tool's error envelope does not cover.
The one writer of the directory, `_save_poses`, creates it now - after the
document is encoded and outside the commit, so a refusal names the directory
rather than the temp file it never got to write.
