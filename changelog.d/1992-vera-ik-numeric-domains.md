### Fixed: the VERA IK path refuses a `translation_scale` or `ik_smoothing` it cannot honor

`VeraPolicy` turns the server's end-effector delta chunk into joint targets
inside `get_actions`, and two caller-supplied numbers shaped every target it
produced without a domain between them and the arm. Both are *applied* rather
than forwarded, so nothing downstream refused an unusable value — it became a
plausible-looking chunk of joint targets instead of an error.

`translation_scale` (`VeraPolicy.set_ik_target`, and
`decode_vera_delta_chunk_to_targets` directly) multiplies every translation
delta. `0` discarded the translation half of every action and returned a
rotation-only chunk under a `tracking_error` reporting a perfect solve; a
negative value inverted it; and `nan`/`inf` made **every** returned joint target
non-finite, along with the `tracking_error` that would otherwise have reported
it — `send_action` then refused each target for being non-finite, which reads as
a wrong-embodiment action-key mismatch rather than as the scale that caused it.
It now takes the same positive-finite domain as the two sibling action
multipliers, `SimEnv.action_scale` and `WBCConfig.action_scale`.

`ik_smoothing` (`VeraPolicy(...)`) weights the previous joint target in the EMA
`target = (1 - alpha) * solved + alpha * previous`, and its own comment already
recorded the interval it needs — `alpha in [0,1); 0 disables` — without
enforcing it. `1.0` (and `True`, which is `1.0`) made every commanded target the
previous one, so the arm froze at the pose the first solve produced and the rest
of the chunk was discarded; above `1.0` the weight on the IK solution turned
negative and the targets diverged away from it, measured at `-5.9x` the solved
joint travel for `1.5` and `-35.3x` for `2.0`; and a negative or `nan`
coefficient failed the `alpha > 0` test the blend is gated on, so the damping the
caller asked for was silently never applied. It is now checked against that
interval before any config or server work.

A structural test asserts every public VERA surface reading either knob routes
through a domain, so a third one cannot be added without one.
