### Tests: the render-gating rule reads a camera image out of an observation as the GL dependency it is

`tests/test_mujoco_render_assertions_are_gl_gated.py` keeps a read that needs an
offscreen GL context from reaching the suite ungated, so that a headless host
reports a missing context rather than a bare failure about the contract under
test. It matched one spelling of that dependency, `render*(...)["status"] ==
"success"`, and a camera indexed out of `get_observation` is a second one.

That spelling degrades worse. `_render_cameras` logs a per-camera failure at
debug level and omits the key it could not produce, so the read raises
`KeyError: 'camera1'` - naming neither GL, nor the contract, nor the fact that a
render was attempted.

The rule now reads both. The discriminator for the new spelling is the key, not
the call: a subscript counts when its key is a camera the module registers, or
the free view every world registers, and when the call did not pass
`skip_images=True`. Sixteen modules index an observation and two of them name a
camera; keying on the call would have reported seven reads across five modules
whose keys are joint angles and base poses.
