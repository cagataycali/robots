### Fixed: no mesh wire knob is writable from the dashboard's env editor

The gate bearing fence on `POST /api/config` now covers the whole
`STRANDS_MESH_` vocabulary. Before, it named three mesh prefixes that matched
one real variable between them, so `STRANDS_MESH_LOCAL_DEV` (which alone
defaults the wire to `auth_mode=none` and stands in for the insecure
acknowledgement) and `STRANDS_MESH_MULTICAST` were page writable through the
env editor: one settings save turned mesh auth off for the next session and
for every child that read the env file. Both stay visible in the drawer, read
only. Every mesh knob is set on the host. (f007)
