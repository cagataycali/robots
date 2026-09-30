### Fixed: `open_viewer()` on macOS names `mjpython` and the command to run

MuJoCo's passive viewer runs on macOS only under `mjpython`, the launcher the mujoco wheel installs next to `python`. The refusal now says so with the exact command and points at `render()` for frames without a window; the MuJoCo page says the same next to `open_viewer()`. (#4169)
