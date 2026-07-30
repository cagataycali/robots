### Fixed: a camera pixel dimension one backend refuses is now refused by all of them

`width` / `height` reach two surfaces on every simulation backend - the
`add_camera` that fixes a camera's resolution, and the render family
(`render` / `get_frame` / `get_camera_params`) that can override it per call.
MuJoCo validated both through `_validate_render_dims`; Newton and Isaac
validated neither and coerced with a bare `int(...)`, which refuses nothing
useful:

```python
sim.add_camera(name="wrist", width=0, height=-4)
# status="success", camera registered 0 x -4 - the failure surfaces at the
# first render, far from the call that caused it

sim.add_camera(name="wrist", width="big")
# ValueError: invalid literal for int() with base 10: 'big'
# raised straight through the structured tool-result contract
```

Three more classes of the same gap: `int(width or default)` read a falsy `0`
as *omitted*, so a caller who asked for an impossible resolution was handed the
default and told it was what they requested; `2.7` and `True` were silently
truncated to 2 and 1, while Newton's success text echoed the caller's raw value
and so named a resolution that was never registered; and on Isaac a stored
negative width was multiplied back out by the DLSS upscale
(`scale = _MIN_RENDER_PX / w`), giving a native render size of `640 x -76800`
so that every later render of that camera failed on the negative dimension -
one bad configuration call disabled the camera for the rest of the session.

Both backends' `add_camera` and render family now validate on
`strands_robots.utils.positive_count_error`, the domain MuJoCo's floor already
implements: a true `int` (`bool` refused - an `int` subclass whose `True` would
act as a silent 1) that is `>= 1`. A pixel dimension is consumed directly as an
array or framebuffer dimension, where an integral float raises `TypeError`
rather than being coerced, which is why it shares that domain rather than the
looser `positive_whole_number_error` used for frame rates. `None` still means
"take the configured default" - membership decides that, not truthiness. The
*upper* bound stays backend-specific, since MuJoCo has an offscreen framebuffer
to overflow and the ray-traced backends do not, so a 5000-pixel-wide Newton
camera keeps working. Valid resolutions behave exactly as before.
