### Tests: the dashboard session-renewal decision is graded, cap included

`dashboard/auth.py` decides how long a browser keeps the right to command real
hardware. Two modules already name `renewal_verdict` / `issue_handoff`, but only
to MEASURE a duration knob's window - `test_dashboard_auth_duration_knobs_are_not_widened`
bisects for the largest renewable age. Neither graded the decision, so every
refusal arm, the half-life gate, the `iat0` fallback and the whole of
`renew_if_due` ran uncovered: 22 of `auth.py`'s 30 uncovered statements sat in
that one family.

Four invariants the code states and nothing held:

* **The cap.** `iat0` is "carried unchanged through every renewal so the absolute
  cap in renewal_verdict() cannot be reset by re-issuing". A session that renews
  forever never needs a passkey again.
* **No downgrade.** A renewal is "never SHORTER than what the client already
  holds: a renewal that shaved time off would be a downgrade the client cannot
  refuse".
* **An unverifiable token is a login problem, not a renewal one**, so
  `renew_if_due` returns `None` rather than re-signing claims it could not check.
* **A handoff never outlives its session**, because it rides in a URL and URLs
  land in history, logs and screenshots.

The centrepiece drives the real surface - `renew_if_due` -> `verify_token` ->
`renewal_verdict` -> `issue_token` - across a simulated clock from sign-in to the
cap, one decision per half token-lifetime, asserting the whole ladder rather than
a single verdict. That shape is required by how the cap works: each renewal is
CLAMPED to `iat0 + max_age`, so no single call ever shows a session running out of
headroom. It lands exactly on the cap, is then refused because extending further
would shave time off the token held, and finally expires there. The ladder's
session signed in half a token-lifetime before the wall clock, because a session
minted at the current second hides a renewal that re-dates from `now`.

Around it: the refusal tables for both verdict functions (each spelling drawing a
refusal an operator can act on - "no expiry to extend" points at the token,
"already expired" points at the passkey), the `iat0` fallback that dates a
session from its token rather than treating a phone's stored token as brand new,
a `SESSION_MAX_AGE` below `TOKEN_TTL` showing the cap is a renewal gate and not a
revocation, four unverifiable tokens, and the handoff's identity carry.

Measured by mutation: eight changes to the behaviours these cells grade, all
eight detected. `dashboard/auth.py`: 30 -> 8 missing, 94% -> 98%; the eight left
are the enroll-token write and the two proxy-peer helpers, outside this family.
No production statement changed.
