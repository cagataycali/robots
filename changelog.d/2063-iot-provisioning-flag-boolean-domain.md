### Fixed: refuse a non-boolean IoT provisioning flag instead of reading it by truthiness

`mesh.iot` has two public entry points carrying `bool` flags, and every one of
them selects a *posture* rather than scaling a quantity. Each was read by
truthiness, so every non-boolean spelling of *off* - `"false"`, `"no"`, `"off"`,
`"0"`, all truthy - selected the permissive branch.

`provision_robot(allow_estop_publish=...)` is a security opt-out: `False` swaps
the robot's certificate onto `strands-robot-no-estop`, identical to the default
policy but without the `AllowSafetyEstop` publish grant, for the case its own
module comment calls the common one - a robot that should obey fleet stops but
never issue one. Measured against a recording IoT client, `allow_estop_publish`
of `"false"`, `"no"`, `"off"`, `"0"`, `1` or `math.nan` all provisioned
successfully onto the grant-bearing `strands-robot` policy, so the opt-out failed
open and the certificate could originate a fleet-wide stop (and arm a Will on
it). The attachment is durable: it persists on the certificate until someone
re-provisions. The flag also had no `Args:` entry, so a caller could not look up
that it takes a boolean.

`bootstrap_account(confirm=...)` is the confirmation gate in front of an
account-wide create, documented as "Must be True to actually create resources".
`confirm="false"` with `dry_run=False` left `not confirm` False and entered the
create path. `dry_run="false"` was the mirror image - the caller asked to leave
preview mode, stayed in it, and was told nothing.

All four flags (`allow_estop_publish`, `confirm`, `dry_run`, `force_update`) now
route through a shared `boolean_flag_error` domain, checked before any AWS call
and before `boto3` is even resolved, so a refused call leaves no Thing, policy or
certificate behind. A flag arrives already typed, unlike an environment variable
whose only shape is a string, so it is checked rather than parsed: parsing would
only move which spellings invert, since `"on"`, `"enabled"` and `"y"` are absent
from every such vocabulary here and would then read as an opt-in while selecting
the restrictive posture. The two declared spellings are unchanged, and the numpy
booleans a comparison produces are normalised to the `bool` the downstream
readers are annotated for.
