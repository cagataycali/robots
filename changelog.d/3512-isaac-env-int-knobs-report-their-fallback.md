### Fixed: an Isaac integer knob that falls back to its default says so

`_env_int` resolves three step counts - `STRANDS_ISAAC_CAMERA_WARMUP_STEPS`,
`SO101_RECORD_CONVERGE` and `SO101_IDLE_CONVERGE` - and substituted the default
for every value it could not honor without a word, while the neighbouring
`_env_float` reports each of its own rejections and states why: a knob whose
effect is only visible several calls away leaves the operator nothing to correct
against when the substitution is silent. A typo (`3O`), a float spelling
(`10.0`) or a non-positive count now logs a warning naming the variable, the
value and the substitute; the accepted domain is unchanged and blank or unset
stays quiet. `STRANDS_ISAAC_CAMERA_WARMUP_STEPS` is the count of render-bearing
steps `add_camera` takes so a new camera's first frame is real rather than the
pipeline's empty buffer, and its documentation says to raise it on a slow GPU -
so an operator who did and mistyped it got an empty first frame with nothing
naming the knob. Pinned by
`tests/simulation/isaac/test_env_int_knobs_report_their_fallback.py`.
