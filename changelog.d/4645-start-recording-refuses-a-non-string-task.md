### Fixed: `start_recording(task=...)` refuses a task that is not a string

`start_recording` stored its `task` as given and reported `status="success"`.
The value seeds the recorded `task` column for every frame a rollout does not
label itself, so a truthy non-string (`123`, `["pick", "cube"]`) was handed to
the dataset as the frame label, and a falsy one (`None`, `0`, `False`) fell through to `"untitled"`.
It is now refused before anything is armed, with
`start_recording: 'task' must be a string, got <type>.` - the same rule
`run_policy` already applies to `instruction`, on every simulation backend.
