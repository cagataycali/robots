### Tests: the microduck wrong-width premise class holds on every numpy the project declares

`pyproject.toml` declares `numpy>=1.21.0,<3.0.0` and the lock resolves 2.2.6.
numpy removed `np.cross` on 2-element vectors in **2.5.0**, inside that range,
and five cells of
`tests/policies/microduck/test_microduck_observation_refuses_a_wrong_width_base_block.py::TestThePremisesTheDefectRestedOn`
pinned the planar reading it removed. Measured on the same tree with nothing but
the numpy version changed: 2.2.6 and 2.4.0 green (503 passed), 2.5.0 five
failed. CI installs the lock, so nothing in the gate could see it.

The cells now grade whichever behaviour the installed numpy implements, decided
by a probe of `np.cross` rather than by a version comparison, so no band is
skipped and the probe cannot go stale. What they state is the band-independent
invariant the guard rests on: an unguarded short `base_quat` is never refused
**by name**. Below 2.5 it is answered, as exactly the quaternion with its `z`
dropped; from 2.5 it raises out of `np.cross` with a message naming neither the
key, nor its width, nor the caller who supplied it. `_require_base_block` is
what attributes it on either side, so it is not a guard a newer numpy makes
redundant.

The tilt and norm figures both docstrings quoted were also measured before
`quat_rotate_inverse` began normalising the orientation it is handed. The wrong
reading is 8.1 degrees off for the small-yaw pose and 28.0 for the roll-then-yaw
one, at **exactly** unit length rather than 0.991 and 0.935 - a larger error
with no trace left in the magnitude at all. The corrected figures make the
"no norm this module reads can judge a width" premise stronger and retire a
`norm_tolerance` parameter that could no longer fail.

Production behaviour is unchanged: the width guard was correct on both bands
before this change and still is.
