### Fixed

- `simulation`: the MuJoCo backend's `list_robots` reports each robot's **live**
  base position, read from `mjData`, instead of the `add_robot(position=...)`
  request the scene record holds. `position` offsets the model's own authored
  root pose rather than replacing it, so 29 of the 51 single-root robots in the
  built-in registry were listed at a place they had never stood (a `jvrc` asked
  for `z=0` has its pelvis at `z=1.4`), and 32 of the 63 have a floating base,
  so any robot that walked, drove or fell kept reporting its spawn pose - 0 mm
  of displacement for the motion a locomotion rollout is judged on. `add_robot`
  already reported the measured placement, so the two calls had contradicted
  each other for the same robot in the same session. A model with several roots
  has no one base pose to measure and now says the number it reports is the
  requested attach frame.
