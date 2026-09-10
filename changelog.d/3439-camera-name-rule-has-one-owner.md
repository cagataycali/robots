### Fixed: `add_camera` names the same cause for the same camera request on every backend

The two halves of the camera-name rule - `entity_name_error` (a value that cannot
be a registry key) and `reserved_camera_name_error` (a `str` the backend's own
`render` / `get_frame` resolve to the free camera) - were applied separately at
each `add_camera`, and the sites disagreed about where the name rule sits
relative to the *value* rules. MuJoCo judged the name first; Newton judged it
after `position`, `target`, `fov` and the pixel dimensions. Both refused the same
request, which is all the cross-backend parity pin compared, while naming
different causes:

```
add_camera("default", fov=0.0)              mujoco: 'default' is reserved
                                            newton: 'fov' must be in (0, 180)
add_camera("default", position=[nan, 1, 1]) mujoco: 'default' is reserved
                                            newton: 'position' must contain finite numbers
add_camera("free", width=0)                 mujoco: 'free' is reserved
                                            newton: width must be a positive integer
```

A reserved name is the one fault no change of value can clear, so a caller
following Newton's message fixed a value and learned the name was unusable on the
round trip after it. `reserved_camera_name_error` already documented its
dependence on the order ("that guard runs first at every call site"), an
assumption no single site owned.

The rule and its order now live in `strands_robots.utils.camera_name_error`,
which every backend's `add_camera` reads. Isaac passes
`routes_free_camera_tokens=False`, so its documented `"default"` signature default
keeps working and that divergence is a stated property of the call rather than a
guard one site omits. No MuJoCo message changed.
