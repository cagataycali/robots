### Fixed: `CompositePolicy` refuses a joint-group or observation-key list it cannot route with

`CompositePolicy(lower_joints=..., upper_joints=..., lower_obs_keys=..., upper_obs_keys=...)`
turned each list into a `set()` with no domain, so a single name passed as a bare
string was read one member per character. `lower_joints="left_knee"` left the lower
policy owning no joint of the robot: every command it emitted was dropped from the
merged chunk, under a successful call and with nothing logged - the silent drop the
class documents as "a fall, not a warning". Two bare strings produced an empty action
dict, and because the disjointness check ran on the character sets it could report an
overlap for two names that share no joint at all. A mapping had its values discarded,
and a non-string or repeated entry claimed nothing.

All four lists are now validated on the shared `name_list_error` domain before the sets
are built, gated on a truthy value so `None` ("no explicit group") and `[]` ("claim
nothing") keep their meanings.
