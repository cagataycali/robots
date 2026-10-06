### Fixed: the Feetech arms page names the native driver as the default

`docs/learn/hardware/feetech-arms.md` said `Robot("so101", mode="real", port=...)`
builds the lerobot driver. `resolve_driver` picks the native `FeetechDriver`
for every robot the page compares (so100, so101, lekiwi, hope_jr,
open_duck_mini), so the quickstart comment and the comparison table now say
`driver="strands"` is the default and `driver="lerobot"` is the opt-in. A docs
test now grades any "<driver> driver, the default" comment or
`driver="..."` (default) table header against `resolve_driver`.
