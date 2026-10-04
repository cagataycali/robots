### Fixed: an unknown-keyword refusal names the keyword you meant

`Teleoperator("so101_leader", prot=...)`, `Robot("so101", mode="real", prot=...)`
on `driver="lerobot"` and on `driver="strands"` now answer
`Did you mean: 'prot' -> 'port'?` beside the accepted roster, as the camera-option
refusal already did. One helper, `strands_robots.utils.did_you_mean`, writes the
clause for all four.
