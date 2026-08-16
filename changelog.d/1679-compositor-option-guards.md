### Fixed: HybridCompositor refuses render options it cannot honor

Every option the hybrid compositor takes was coerced rather than checked, so a
value with no usable interpretation still returned a frame - just not the one
the caller asked for. `depth_epsilon` (the "this pixel has no geometry" depth
threshold) was stored with a bare `float(...)`:

```python
frame = HybridCompositor(sim, depth_epsilon=float("nan")).render("look")
frame.foreground_mask.mean()   # 0.0000 - the whole robot is gone
```

`fg_depth > nan` is false for every pixel, so the entire simulation foreground
read as empty and the composite was the background alone, with no diagnostic.
`inf` did the same; a negative threshold did the opposite, admitting the no-hit
pixels (Isaac's RTX annotator reports them as `0`) that the parameter exists to
exclude, painting simulation sky over the background.

`feather_pixels` was clamped with `max(0, int(value))`, so `-5` and `2.7`
silently became `0` and `2` - the first disabling the seam blend the parameter
exists to apply. And `render(width=..., height=...)` read its size with
`width or self.default_width`, so a supplied `0` read as "not supplied" and the
frame came back at the default size, even though that same `0` is refused when
it reaches `get_camera_params` as `default_width`.

All four options are now validated where they are supplied - pixel counts
through the shared `positive_whole_number_error` domain, `depth_epsilon` as a
finite distance in meters `>= 0` (`0` remains valid: "only an exactly-zero
depth is no geometry"), and the render size read by membership rather than
truthiness - so a `default_width` mistake names `default_width` instead of
surfacing later as a render error. Integral options are normalized to plain
`int`, so a `np.int64` size no longer leaks into the requested camera size.
