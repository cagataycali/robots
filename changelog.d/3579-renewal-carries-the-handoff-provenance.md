### Fixed: a renewed dashboard session keeps the `via` marker that says where it came from

`auth.issue_token` writes a `via` claim that "marks how the token was minted
(e.g. "handoff") for later forensics", and `auth.issue_handoff` is its only
writer: the short-lived token that rides in a URL to move a signed-in session to
a phone. `auth.renew_if_due` re-minted the session without `via`, so the marker
was erased and a session that began as a URL handoff became indistinguishable
from one signed by a passkey.

It was erased at the first use rather than eventually. A renewal is due once
`now >= exp - TOKEN_TTL / 2`, and that threshold is measured against the
configured `TOKEN_TTL` (a day) rather than the lifetime the presented token
actually has. A handoff token lives for `HANDOFF_TTL` (five minutes), so at the
moment it is minted it is already 42,900 seconds past the threshold - its first
renewal is its first use. `issue_handoff` documents renewal as the expected next
step for exactly that token, "carrying the session's identity (sub/name/iat0) so
renewal caps survive the copy", so the one claim recording where the session came
from was the one claim the copy did not survive.

`renew_if_due` now carries `via` the way it already carries `iat0`, which the
module describes as "carried unchanged through every renewal": both claims
describe the ORIGINAL sign-in, and a renewal is the same session continuing
rather than a new one. A session that never came from a handoff still gains no
marker.

The verb had no test of any kind before this - it is the middleware helper the
dashboard's server slice will call, and nothing in the package, tests or docs
referenced it - so the arithmetic that was already right is pinned alongside the
fix: a renewal never shortens what the client already holds, never passes the
absolute session cap, carries the original sign-in forward, and is refused for a
token that is empty, malformed, signed with another secret, already expired,
still fresh, or past its maximum age.
