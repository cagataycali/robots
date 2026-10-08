### Fixed: a Newton recording renders each declared camera once per frame and writes only those

Newton's recording hook rendered every declared camera again on every recorded
frame, even when the rollout's observation already carried the image, and
copied every array of the observation into the frame - so a recorded step paid
two renders per camera and a camera outside `start_recording(cameras=...)` was
written under its scene name beside the schema. Newton, mjlab and Isaac (its
`run_policy` hook and `run_multi_policy`) now split an observation through one
rule: an image of a declared camera goes under its column, any other camera is
dropped, and only a declared camera the observation lacks is rendered.
