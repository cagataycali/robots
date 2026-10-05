### Fixed: a recording target that is, or holds, the working or home directory is refused

`start_recording(repo_id="", overwrite=True)` read the empty id as the local path
`.` and replaced it with `shutil.rmtree`, deleting the caller's working
directory; `repo_id="."`, `root="."` and `root=".."` did the same. The writer's
directory resolver (`resolve_dataset_dir`, shared by `start_recording`,
`DatasetRecorder.create`/`resume` and teleop recording) now refuses an empty or
non-string id and any directory that is, or contains, the working or home
directory, and `start_recording` resolves it before the session arms, so the
refusal comes back as an error naming `repo_id` instead of a raised exception
with the recording flag left set.
