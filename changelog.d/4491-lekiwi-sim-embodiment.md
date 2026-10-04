### Fixed: lerobot_local ships a `lekiwi_sim` embodiment for the simulated LeKiwi

`embodiment="lekiwi"` is lerobot's hardware driver type and resolves to `lekiwi_real`
(`arm_*.pos`, `x.vel`/`y.vel`/`theta.vel`), which the MuJoCo LeKiwi neither reports nor
actuates, and no sim entry existed to name instead. `lekiwi_sim` declares the nine sim
joints as state and the nine actuators (wheel velocities, arm positions) as actions. A
declared embodiment that binds none of the observation's keys now raises a `ValueError`
naming the embodiment that does, instead of failing inside the model on
`KeyError('observation.state')`.
