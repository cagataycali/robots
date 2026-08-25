### Fixed: the terrain seed names exactly one value-noise stream

`generate_heightfield` documents itself as "deterministic given `(kind, resolution, seed)`", and
the module docstring promises that a benchmark evaluating a policy on `terrain="rough"`
"regenerates the identical field on every reset". `resolution` was measured against the shared
positive-discrete domain; `seed` was handed straight to `random.Random`, which does not seed from
the value it is given - it seeds an int from `abs(value)` and anything else from `hash(value)`. So
the documented triple was neither injective nor total.

Three distinct seeds drew one field. `seed=-1` shares a stream with `seed=1` because the sign is
discarded, and `seed=True` shares it too because `bool` is an `int` subclass - the same collision
`derive_variant_seed` already names for the other seed in the package that is spread into a stream
key. A curriculum stepping the seed across resets to draw fresh ground silently re-drew ground it
had already evaluated on, and nothing reported that the two resets shared terrain.

`seed=float("nan")` was worse than ambiguous: it was irreproducible. `hash(nan)` has been derived
from the object's identity since Python 3.10, so two `float("nan")` seeds draw two different fields
*within a single process* - the one input for which the module's headline promise is false rather
than merely surprising. `seed=None` was accepted too, and seeds `random.Random` from the OS entropy
pool, so it drew a fresh field on every call. `seed=2.5` and `seed="1"` were accepted outright, the
same fractional and string axes the `resolution` domain already closed.

The seed is now measured against `non_negative_whole_number_error`, the shared domain
`derive_variant_seed` applies to the same quantity. Non-negative closes the `abs()` alias, whole
closes the `hash()` path that admitted `nan`, `inf`, `2.5` and `"1"`, and the domain's boolean
refusal closes `True`/`False`. An integral float such as `2.0` is still accepted, because it hashes
to its int and so names the same stream rather than a second one. Measured over the production
resolution: before, `seed=1`, `seed=-1` and `seed=True` all produce field md5 `ebb516ac` and render
byte-identically; two `nan` seeds produce `07b782df` and `40ab6cf9` and differ in 48990 of 69000
render pixels. After, `seed=1` still produces `ebb516ac` and renders byte-identically, and the
ambiguous spellings raise `ValueError`.

Scoped two ways, both pinned. The domain is applied only on the `"rough"` branch, the only kind
drawn from an rng: `_stairs`, `_pyramid` and `_slope` take no seed at all, so refusing them for one
would refuse a caller a value the requested kind cannot act on. The split is read off the
generators' signatures rather than hardcoded, so a kind that later gains an rng cannot inherit the
exemption. And no upper bound is imposed on top of the shared domain, because `random.Random`
consumes an arbitrarily large int directly - the narrower seed domain the simulation backends apply
is capped to NumPy's 32-bit stream range and would refuse a value that is usable here.
