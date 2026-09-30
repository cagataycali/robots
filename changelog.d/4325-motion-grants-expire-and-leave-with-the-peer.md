### Fixed: a motion grant expires and leaves the fleet with its peer

The one-shot grant the dashboard hook deposits when an operator approves a
motion sat in a process-global set until a byte-identical call spent it, so a
yes given in the morning was spendable that evening, for a robot that may since
have left or been swapped under the same name (f026, CWE-613). Every grant now
carries its deposit time and target: it expires after
`STRANDS_DASH_MOTION_GRANT_TTL_S` (900 s by default, read at spend time, with
unusable values falling back rather than removing the bound), the mesh bridge
forgets a peer's grants when it ages out of the fleet snapshot or the mesh is
re-pointed (`grants_forgotten` in the trail), and `pending_grants()` lists what
is outstanding without exposing the keys.
