### Fixed: an append reopens the dataset its `repo_id` created

`DatasetRecorder.create` resolves the dataset directory once, through
`resolve_dataset_dir`, and forwards the result to LeRobot as an explicit `root`.
`resume` - the append entry point, and the only writable way into an existing
dataset - forwarded the caller's `root` unresolved. An absent one stayed absent,
and `LeRobotDataset.resume` refuses that outright: the directory it would derive
for a writer is the revision-safe Hub snapshot cache. So the append was
unreachable on exactly the arguments `create` accepts, and the refusal asked for
a directory this repo already names - the one the dataset was written to.

Measured against lerobot 0.6.2 on a MuJoCo scene:
`start_recording(repo_id="probe/append_two_episodes", root=None, overwrite=True)`,
a 30-step rollout, `stop_recording()`, then the same call again with the default
`overwrite=False`. The first session wrote one episode with every call
`status="success"`; the second returned `status="error"` - "Dataset init failed:
resume() requires an explicit 'root' directory ..." - after resolving that same
directory to decide an append was wanted and stashing it as
`last_dataset_root`; `verify_dataset_episodes(expected=2)` then reported 1
episode / 30 frames. With the resolution in place the script records 2 episodes /
60 frames and every call reports success. All three backends' `start_recording`
route their append through this method, so all three were affected.

An explicit `root` is unchanged: it was, and still is, used verbatim. Every
existing fake dataset class in the suite accepts `root=None` silently, which is
why none of them observed this; the double in
`tests/test_dataset_append_reopens_what_the_repo_id_created.py` reproduces the
one property that decides the outcome - a writer refuses an absent root.
