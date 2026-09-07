### Fixed: a per-call Kimodo sampler knob is graded, not converted

`KimodoPolicy.get_actions` read its documented `diffusion_steps` and
`guidance_scale` overrides straight out of `kwargs` and converted them with
`int()` / `float()`, which cannot refuse a value - only reinterpret it. An
override reaches the sampler and the buffered-motion key without passing
through `KimodoConfig`, so of 18 values the config field refuses, none were
refused per call: 12 were admitted (`diffusion_steps=True` denoised once,
`2.7` became 2, `1000` escaped the field's ceiling, `guidance_scale=nan`
reached the sampler) and 6 escaped as a `TypeError`/`OverflowError` naming
neither the surface nor the knob. Ten of the admitted values also discarded
the motion being tracked to re-sample with a number the caller never named.

Both overrides now consult the same domain their config field does - the new
`diffusion_steps_error` owns the composite step-count domain (the shared whole
number rule plus this provider's ceiling) - and are checked before the key is
built, so a refused override leaves the held motion and the frame cursor
exactly as they were, as the sibling `seed` override already did. Values the
fields accept, including an integral float and a NumPy scalar, are unchanged.
