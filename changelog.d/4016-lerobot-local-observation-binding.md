### Docs: `lerobot_local` observation binding is its own page

`docs/policies/lerobot-local.md` documented the provider surface (install,
parameters, model caching, RTC) and the whole observation-binding reference
(normalization stats, the unit frame those stats are recorded in, the joint keys
that compose `observation.state`, camera routing and the pre-flight check) on one
2,801-word page. The binding reference moves to
`docs/policies/lerobot-local-observations.md`, leaving 1,301 and 1,461 words, and
the camera-naming / MolmoAct2 links and the `obs_rename` pre-flight anchor point
at the page that now holds each.
