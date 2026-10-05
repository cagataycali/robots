### Fixed: the docs say `STRANDS_ROBOT_MODE` is read only under `mode="auto"`

The robots and architecture concept pages said the variable decides when no
`mode` is passed. `mode` defaults to `"sim"`, so a bare `Robot("so100")` never
reads it; only `mode="auto"` does. The pages and the `strands_robots.robot`
docstrings now say so, and a test pins that `STRANDS_ROBOT_MODE=real` leaves a
bare `Robot()` in sim.
