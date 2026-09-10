### Fixed:
`MockPolicy` now binds the sim's compiled model (`set_sim_context`) and swings each actuator through the middle half of its own ctrlrange, instead of an unit-less `0.5 * sin` that MuJoCo clamped on the so100's `Pitch`, `Elbow` and `Jaw`. A dataset recorded from the mock policy therefore stores actions the robot actually executed; unbound use (no sim) is unchanged.
