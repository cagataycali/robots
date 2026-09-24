### Fixed: the episode-contract and collection-loop tests record action-only datasets

`tests/simulation/test_run_policy_episode_contract.py` and
`tests/simulation/test_collection_loop_resets_between_episodes.py` read episode
indices, frame counts and each episode's first `observation.state` out of the
datasets they record; no cell reads a pixel. Their `start_recording` calls kept
the world's free camera, so every rollout rendered it through OSMesa at each
control step and encoded a video per episode flush - at `n_episodes=20` that was
20 s of a 21 s cell, and 12 s of each 13 s collection-loop cell, on a 2-vCPU
runner (#3869). The four `start_recording` calls now pass `cameras=[]`, the
action-only form the recorder documents; the two files run in 6 s instead of
145 s with every assertion unchanged.
