### Fixed: the episode-boundary status test records an action-only dataset

`tests/simulation/test_recording_episode_boundary_is_told_to_the_agent.py::test_status_reports_saved_episodes_beside_the_open_buffer`
asserts frame and episode counts and reads no pixel, yet its recording kept the
world's free camera, so four `run_policy` rollouts rendered it through OSMesa at
every control step and encoded a video per episode flush. Measured on a 2-vCPU
runner that was 39 s of the cell's 41 s, the second-largest single cell in the
suite per #3869. The two `start_recording` calls now pass `cameras=[]`, the
action-only form the recorder already documents; the cell runs in 4 s and every
assertion is unchanged.
