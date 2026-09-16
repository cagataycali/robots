### Fixed

- `lerobot_local`: the state-key remedy no longer recommends an `embodiment=`
  that this policy already rejected. A declared embodiment is applied as a whole,
  so an `obs_rename` naming an image feature the checkpoint does not declare
  discards the state binding too; the auto-generated `joint_0..joint_N` ordering
  then survived and both state-key guards handed back the embodiment the caller
  had just passed. They now read the same `_embodiment_config_failed` the
  missing-postprocessor warning consults, and point at the camera routing that
  makes the declared embodiment validate.
