### Fixed: a recorded dataset lands in the directory `create` prepared for it

`DatasetRecorder.create` resolved the dataset directory twice. It resolved one
through `resolve_dataset_dir` and handed it to `_prepare_create_target`, which
inspects that directory and - under `overwrite=True` - deletes it; then it called
`LeRobotDataset.create` with the caller's raw `root`, letting the writer resolve a
directory of its own. The two rules differ: `resolve_dataset_dir` reads a
`repo_id` that is itself a path (no `owner/name` slash, or `./`-prefixed) as a
local directory, while LeRobot resolves any absent root to
`$HF_LEROBOT_HOME/{repo_id}` whatever the id looks like.

Measured end to end against lerobot 0.6.2 -
`start_recording(repo_id="sim_recording", root=None, overwrite=True)`, a 45-step
MuJoCo rollout, `stop_recording()`, every call `status="success"`: an unrelated
`./sim_recording` holding one file was deleted, the dataset was written under
`$HF_LEROBOT_HOME/sim_recording`, and `verify_dataset_episodes(expected=1)` then
reported `status="error"`, since the facade records the prepared path as the
dataset root. With a dataset already at the writer's directory, `overwrite=True`
could not overwrite at all: the wipe missed it, and LeRobot's own
`mkdir(exist_ok=False)` raised the bare `FileExistsError` that
`_prepare_create_target` exists to replace with a message naming `overwrite=True`
and `resume()`.

`create` now resolves once and forwards the result as an explicit `root`, so the
directory it inspected and wiped is the directory the dataset is written into -
which is what `docs/recording.md` already claimed by calling
`resolve_dataset_dir` the one owner of those rules. An `owner/name` id is
unchanged (both rules already agreed on `$HF_LEROBOT_HOME/{repo_id}`), and
`resume`, which resolves and prepares nothing, is untouched.
