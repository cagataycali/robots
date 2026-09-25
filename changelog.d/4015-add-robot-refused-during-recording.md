### Fixed: `add_robot` is refused while a dataset recording is open

A recorder declares its features once, from the robots attached at
`start_recording`, and a LeRobotDataset cannot gain a column afterwards, so a
robot added into a live recording had nowhere to be written. Against a
camera-only schema the next rollout died inside lerobot with a feature mismatch
after 0 frames, leaving an empty dataset shell on disk; against a 6-wide so101
schema a 9-joint panda rollout raised nothing and saved frames of the frozen
robot's zeros rather than the moved robot's state. `add_robot` now refuses on
every backend while a recording is live, naming the frozen schema and the
`stop_recording` / `add_robot` / `start_recording` order that records both
robots.
