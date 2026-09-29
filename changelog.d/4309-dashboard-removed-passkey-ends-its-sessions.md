### Fixed: removing a dashboard passkey ends the sessions minted for it

`auth.verify_token` checked a session token's signature and expiry and nothing
else, and `delete_credential` rotated nothing, so a token issued for a removed
passkey kept working until its `exp`, could renew itself and could mint a
handoff (f016, CWE-613). `verify_token` now also requires the token's `sub`
(the credential id every issuer stamps) to name a passkey that is enrolled at
that moment, from the store it already loads for the secret, and refuses a
revoked session with its own sentence so it reads differently from an expired
or forged one in the log. Renewal, handoff and the middleware check inherit it;
a dashboard with no passkey honours no session; the duration knobs are unchanged.
