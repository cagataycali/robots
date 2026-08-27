### Tests: the raw-motion resample onto the control rate is covered

`MotionPlayer` takes a raw ProtoMotions `.pt` as a first-class source and
resamples it onto the tracker's control rate, and that whole path was
unexercised (#2718). The existing `.pt` coverage writes a *cache-shaped*
payload, which returns from the loader before the resampler runs, so both raw
layouts - a packed library sliced out of a concatenated buffer by
`length_starts`, and a single motion declaring its own `fps` - the rate
conversion, the rotation interpolation and the unrecognised-layout refusal had
no test.

`tests/policies/protomotions/test_raw_motion_is_resampled_to_the_control_rate.py`
keys every fixture row to its global source frame index, so an assertion names
the source row a resampled frame came from rather than checking that it looks
plausible. The source rate is 10 Hz and the control rates are exact multiples of
it, so each resampled frame either lands on a source instant (and equals that
row exactly) or at a known fraction between two (and equals their blend
exactly), which needs no tolerance for the five linearly interpolated channels.
Rotations are held to the two properties that separate spherical interpolation
from the alternatives by measurement: unit length (a blend taken on the line
measures `0.9239` for the fixture's 90-degrees-apart neighbours) and a constant
angular rate (a normalised line blend alternates `21.598 deg` / `23.402 deg`
against a uniform `22.5 deg`).

This takes `strands_robots/policies/protomotions/motion_utils.py` from 64% to
100% statement coverage - 61 uncovered statements to 0 - with no production
change.
