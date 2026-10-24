### Fixed: an agent sees the shipped policy presets before its first call

The `policy` argument of `robot.tool` `run` and `sim.tool` `run` described itself as "a policy spec, e.g. a LeRobot
checkpoint id", so an agent learned `holosoma` and `microduck` only from the error of a wrong guess. The schema now
names every preset in the registry and every spec form that crosses a wire (preset with a query, `zenoh://`, a Hub
repo, a local checkpoint, a recorded dataset). A preset added to `presets.json` appears without touching the schema.
