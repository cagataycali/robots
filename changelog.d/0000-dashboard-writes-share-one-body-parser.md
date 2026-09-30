### Fixed: a dashboard write with a body that is not JSON is a 400, not a 500

`POST /api/sim`, `/api/sim/{id}/joints`, `/api/config`, `/api/consent`,
`/api/consent/revoke` and `/api/agent/reset` now read their body through the
same parser the auth, mesh and settings routes use. A body that does not parse
is a 400 that says so (it was a 500 with a `JSONDecodeError` traceback in the
log), a body that is not `application/json` is a 415 like every other write
(a `text/plain` write is the no-preflight request a cross-site page can send),
and an empty body is `{}`. The dashboard's own page always sends JSON, so it is
unaffected.
