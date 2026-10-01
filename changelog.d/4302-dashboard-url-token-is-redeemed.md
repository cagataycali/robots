### Fixed: a `?token=` in the dashboard URL is redeemed against the backend, never simply believed

The dashboard wrote whatever `?token=` held straight into the stored sign-in on page load, so
any string in a link signed the operator out and a token the backend would verify signed them
in as someone else, silently. The offered token is now held in memory until one probe of
`GET /api/auth/status` on the backend the page is already configured for answers
`authenticated: true` for it; it is dropped without a probe when it is not a hand-off token,
has lapsed, arrives beside a `?backend=` that moves the page, or when the browser already holds
a valid sign-in, and a refusal is said on the sign-in screen (finding f019).
