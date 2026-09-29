### Fixed: a policy rate the physics step cannot divide is reported

`PolicyRunner` rounds the control period to a whole number of physics steps; with Isaac's default `physics_dt=1/120` a 50 Hz policy silently ran at 60 Hz. It now logs a warning naming the effective rate and a `physics_dt` that divides the period.
