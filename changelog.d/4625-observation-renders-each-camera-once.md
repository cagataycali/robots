### Fixed: a MuJoCo observation renders each camera once, whatever it is called

A robot's camera appears in `get_observation()` under two keys, its namespaced
model name (`lekiwi/front`) and the short name `add_robot` registers (`front`).
Each key was rendered separately, so one camera was drawn twice per observation
and the two keys could carry different pixels. Each camera is now rendered once
and every key that names it carries the same frame (in its own buffer). On
`stretch3` an observation now takes 6 renders instead of 11.
