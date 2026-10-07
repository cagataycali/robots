### Fixed: `add_object(size=...)` builds the same object on Newton, mjlab and Isaac as on MuJoCo

`size` is the full extent in metres on every backend: a box's edge lengths, a
sphere's diameter, a cylinder's or capsule's `[diameter, unused, length]`.
Newton and mjlab used to hand it to the engine as half-extents and radii, so the
README's `size=[0.05, 0.05, 0.05]` cube was 10 cm wide there, and Isaac read a
sphere's `size[0]` as its radius. Newton and mjlab now also refuse a vector the
MuJoCo backend refuses (too few components, a non-positive extent) instead of
padding it. Callers who wrote Newton or mjlab sizes as half-extents, or an Isaac
sphere as a radius, double those values.
