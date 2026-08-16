### Fixed: an unrecognized `STRANDS_ISAAC_*` switch value is refused rather than read as off

`STRANDS_ISAAC_HEADLESS` and `STRANDS_ISAAC_RTX_PATHTRACING` are documented as
two-sided switches -- "Truthy (`1`/`true`/`yes`) forces headless; falsy forces a
window" -- but only the truthy side was enumerated, and "falsy" was implemented
as everything else. A spelling that means *on* therefore selected *off*:
`STRANDS_ISAAC_HEADLESS=on` and `=enabled` both opened a window, so did a
leading space or a trailing newline (the read had no `.strip()`), and so did a
set-but-empty value -- which is what an undefined `${{ vars.* }}` interpolation
in a GitHub Actions `env:` block produces. 9 of 17 measured spellings forced a
window without meaning off, on the one flag whose purpose is to keep Isaac Sim
off a display. The neighbouring pathtracing read differed on the same input: it
silently ignored an unrecognized spelling rather than inverting on it.

Both reads now go through one reader over four symmetric pairs -- `1`/`0`,
`true`/`false`, `yes`/`no`, `on`/`off` -- case-insensitive and
whitespace-stripped. Unset or empty means absent and keeps the `IsaacConfig`
field. Any other spelling raises `ValueError` naming both vocabularies and the
reason it refuses instead of choosing a side. Adding `on`/`off` is what keeps
every spelling that already resolved correctly working while the inverting ones
stop; closing the off side is unavoidable, because while any unlisted spelling
reads as off, every spelling that means on reads as off too.

Precedence is unchanged: the environment still outranks the field for these two
switches while the sibling `STRANDS_ISAAC_NUCLEUS_URL` is the other way round.
That contradiction is documented three times and implemented twice, and is
tracked as a contract decision in #2062 rather than settled here.
