### Fixed: every sim tool parameter says which action takes it, and a refused key names the actions that do

Twelve parameters in the MuJoCo tool schema (`duration`, `action_horizon`,
`timestep`, `gravity`, `ground_plane`, `robot_name`, `shape`, `position`,
`mesh_path`, `width`, `height`, `camera_name`) were bare type hints, so an agent
reading the schema had to guess which of the actions each belonged to. Each now
leads with the actions that take it. A key an action does not take is still
refused, and the refusal now also names the published actions whose signature
does take it: `step(duration=2)` answers `... Valid: ['n_steps']. 'duration' is
taken by: run_policy, start_policy.`
