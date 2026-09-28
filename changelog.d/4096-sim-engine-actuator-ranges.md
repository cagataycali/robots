### Added: `SimEngine.actuator_ranges(robot_name)`

Returns each range-limited actuator's `(low, high)` keyed like
`robot_action_keys`; MuJoCo reads `ctrlrange`, other backends return `{}`. The
Yahboom M3 Pro twin's clamp note now reads it instead of the MuJoCo model, so
`strands_robots/drivers/yahboom_m3pro_twin.py` no longer touches `mj_model`.
