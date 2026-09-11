### Fixed: a terrain resolution too small to draw the requested kind is refused

`generate_heightfield` accepted any `resolution >= 2` for every kind, but the
grid a kind needs to draw its shape is a property of the kind. Eleven accepted
counts returned a field contradicting the geometry the module's constants
document, and raised nothing.

Measured against the module's own promises - the normalized `[0, 1]` span, a
stepped kind's declared plateau count, the pyramid's `1.0` centre - `rough` and
`pyramid` at `resolution=2` returned an **entirely flat** field: two box-blur
passes average a 2x2 grid to one value (flat for all 500 seeds tried) and a 2x2
pyramid has a single ring, so both normalized to all-zeros. A caller who asked
for rough ground got the plain floor `terrain=None` gives, scaled to 0 cm of the
`TERRAIN_ELEVATION` they sized the `<hfield>` from; compiled and rendered, the
two requests produced pixel-identical ground. More quietly, `stairs` below
`TERRAIN_STAIR_STEPS` and `pyramid` below `2 * TERRAIN_PYRAMID_STEPS - 1` cannot
hold their plateau count, so the field stopped short of `1.0` - a 4-cell
staircase reached three of five plateaus and topped out at 6 cm where the
documented 2 cm risers say 8 cm.

The `>= 2` floor is now per kind, exported as `TERRAIN_MIN_RESOLUTION`, and a
count below it is refused through this module's `ValueError` contract naming the
kind and the minimum. Every existing property held from these floors upward, so
no field any resolution used to draw changes - the default `TERRAIN_RESOLUTION`
of 40 and the internal `create_world` path are untouched.
