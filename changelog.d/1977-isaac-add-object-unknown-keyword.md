### Fixed: an `add_object` keyword the Isaac backend cannot use is named rather than dropped

All three backends declare the same ten `add_object` parameters. MuJoCo and Newton
declare no `**kwargs`, so Python refuses an unknown keyword with a `TypeError`.
`IsaacSimulation.add_object` alone declared a `**kwargs` sink, read exactly one key
out of it -- `scale`, the documented `size` alias -- and discarded the rest. So
`add_object(name="crate", heigth=0.3)` returned `status="success"` on Isaac having
compiled the default extents and reported them back in the result `json`, while the
same call is a `TypeError` on both siblings. Measured over six unusable keywords
(`heigth`, `positon`, `colour`, `density`, `friction`, `rgba`), every one was
accepted and silently dropped.

`unknown_kwargs_error` exists for exactly this shape and its docstring already
divides `**kwargs` methods into *forwarding* sinks, where dropping is right, and
*discarding* sinks, where it "turns a misspelled or invented parameter into a
successful no-op". The action dispatcher states the same delegation from the other
side -- it skips its own unknown-key check for a `**kwargs` method because "those
methods own the check instead" -- so Isaac's `add_object` was the one action that
skip left uncovered.

The sink stays, because `scale` is a real Isaac-only capability that the docs use;
it now rejects every keyword it does not read, before any prim is constructed, so a
refused call takes no lock, registers nothing and leaves the name reusable.
