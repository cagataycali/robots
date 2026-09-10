### Removed: the `<name>_main` root-body fallback in the predicate DSL

`_body_position` and `_body_quaternion` looked a body name up twice: the name
the spec wrote, then `<name>_main`. That second probe existed for the
procedurally-generated objects of the vendored LIBERO adapter removed in #3056,
whose MJCF root body carried the suffix while the BDDL goal named the object
bare. Both lookups now resolve the name the scene declares, and only that name.

The convention was never applied consistently, which is the reason to remove it
rather than keep it for a scene that might still be shaped that way.
`_geom_belongs_to_body` - the single owner of the body-to-geom mapping, and the
gate every contact-aware predicate runs through - matches `<body>_g`, so
`cube_1_main_g0` is not one of `cube_1`'s geoms. A spec clause naming `cube_1`
therefore read a *position* off `cube_1_main` while `grasped(cube_1)` and
`body_on(..., require_contact=True)` reported no contact at all: two predicates
over one clause disagreeing about which body the clause named, with the
geometric half silently answering for a body the contact half could not see.

Nothing on `main` emits a `_main` root body. `strands_robots/assets` ships no
MJCF (0 files match), no `builtin_benchmarks` spec names such a body, and the
generator that did left with #3056.

A name the scene does not declare now degrades the way every other unresolved
name does - `_warn_unresolved` logs it once and the term becomes a constant -
instead of resolving to a different body that happens to share a prefix. The
`tried` list in that warning drops to the single name asked for, so the log line
names what the spec wrote. A body genuinely called `plate_1_main` still resolves;
the suffix was only ever synthesized, never special-cased away.

`tests/simulation/test_benchmark_predicates.py` pins the resolution by the names
probed rather than by the verdict alone (a recording sim asserts
`probed == [<the one name>]` on both the position and the orientation path), so a
reintroduced fallback fails on the extra probe even where it would return the
same answer. The three tests that pinned the fallback are gone; the two
vendor-named multi-geom tests are renamed for the `<body>_g<idx>` convention
they actually cover, which is shared with strands-native `add_object`
(`<body>_geom`) and stays.
