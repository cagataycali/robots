### Docs: the humanoids page keeps the catalog, each vendor bring-up gets its own page

`docs/robots/humanoids.md` had grown to 1,918 words carrying three hardware
operating procedures beside a generated catalog: the Booster T1's split between
its onboard whole-body controller and the eight upper-body joints a host may
command, the Unitree G1's CycloneDDS bring-up with the `unitree_sdk2py` install
recipe for x86_64 and aarch64, and the Microduck's socket discovery and intent
vocabulary over its on-robot `robotd` daemon.

Each moves to `docs/hardware/` - `booster-t1.md` (611 words), `unitree-g1.md`
(564) and `microduck.md` (603) - where the Reachy Mini bring-up in this same
family already lives. `robots/humanoids.md` (387 words) keeps the catalog, the
humanoid camera-mount recipe and a pointer to each bring-up.

No prose was rewritten: 157 of the 160 moved lines are verbatim, and the three
edited lines are two relative links and one heading promoted from `###` to `##`,
whose `#installing-the-unitree-sdk` anchor is unchanged.

The missing-`unitree_sdk2py` refusal named `docs/robots/humanoids.md`, the page
that no longer carries the install recipe. It now names `docs/hardware/unitree-g1.md`,
and a new cell resolves both the path and the heading that constant spells, so
the next move of that section cannot leave the refusal pointing at a page
without the recipe.
