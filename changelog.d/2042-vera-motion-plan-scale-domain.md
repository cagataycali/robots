### Fixed: `VeraConfig.motion_plan_scale` is refused when no motion plan can be multiplied by it

`VeraConfig` validates both ports and `render_width` on the effective value, at
the one funnel every caller passes through. `motion_plan_scale` arrived through
that same funnel and was checked nowhere, so `0`, a negative, `nan`, `inf`, a
`bool`, a `str` and a `list` were all accepted from a keyword, and `0`, `-1.5`,
`nan`, `inf`, `1e999` and `Infinity` all reached the field from
`VERA_MOTION_PLAN_SCALE`.

Nothing downstream refused them: `_ensure_started` sends the scale to the server
inside a best-effort `except Exception` that logs at INFO and marks the policy
started regardless, so a value `float()` cannot convert was neither applied nor
reported and the rollout proceeded at whatever scale the server already had.

It now takes the same domain as the package's other two scales — a positive
finite number, or `None` to leave the server's scale alone. The best-effort
swallow is unchanged for a genuine transport failure.
