### Fixed: the e-stop override code never travels, and a resume proof clears only the lockout it was minted for

A `resume` command on the command topic carried `STRANDS_MESH_OVERRIDE_CODE` in
cleartext to a topic every robot certificate may read, so one operator resume
taught the code to any admitted peer. `Mesh._dispatch` now refuses every
`resume` command with `resume rejected` and audits why; the operator calls
`Mesh.resume(code)` on its own peer (the dashboard already did), which clears
that lockout and publishes only a proof. The proof key's scrypt salt now mixes
in the fleet namespace (`STRANDS_MESH_NAMESPACE`), and the HMAC also covers the
epoch of the e-stop being cleared - minted by the issuer, carried on
`strands/safety/estop`, supplied by each receiver from its own lockout - so a
captured proof fails in another fleet or against a later lockout, and the
failure counts against the resume throttle. A peer that is not locked audits a
resume as redundant without checking it. An override code must now also
estimate at 64 bits or more (length times the Shannon entropy of its own
characters); a repetitive one is treated as unset and remote resume is refused
there. Fleets upgrade together: a proof from an older peer no longer verifies.
`Mesh._resume_lockout` is renamed `Mesh.resume`.
