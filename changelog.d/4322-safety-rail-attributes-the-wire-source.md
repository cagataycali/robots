### Fixed: the dashboard's safety rail binds an estop or resume to the session that carried it

`MeshBridge._on_safety` folded every `strands/safety/**` body into the fleet
lockout badge as it arrived, naming whoever the body claimed and letting any
resume move a locked fleet to "e-stop?" (f031, CWE-345). The dashboard now
applies the SDK's own `source_zid` binding: a mismatched envelope of either kind
is dropped and written to the trail as `<kind>_refused`; an estop with no wire
identity is still applied, since stopping never gets harder, and its badge
reason says the sender is unverified; a resume the dashboard cannot attribute
leaves the lockout as it was, with the reason on the badge and
`resume_unattributed` in the trail, until a peer proves it clear; an attributed
resume still lands on "unknown", never "clear".
