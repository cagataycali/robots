### Fixed: a dashboard credential store that parses and still cannot serve as one is recovered instead of answering 500

`_load()` repaired a store it could not READ - an `OSError` or unparseable JSON
went to `_preserve_corrupt`, which keeps the bytes and comes up on a fresh
default - but handed back anything that parsed, whatever it held. Every reader
then indexed the document without a second look, so a file that is valid JSON
and not a store put the fault inside whichever route asked. A store with
credentials and no `jwt_secret` raised `KeyError` from `_jwt_secret`: HTTP 500
on every route that verifies a session, for as long as the file sat there, with
no remedy in the response or the log, while the same request WITHOUT a token was
a clean 401. A `jwt_secret` that was empty or not a string was quieter and worse
- every token failed to verify (401 forever) while signing in raised from PyJWT,
so an operator saw "invalid session" and could never obtain a valid one. A
top-level JSON array, string, number or `null`, or a `credentials` value that is
not a list of records each carrying an id, raised `TypeError`/`AttributeError`
out of `auth_enabled()` and `list_credentials()` - the login screen rather than
one guarded route.

All of it is the same condition as an unparseable file, one step later, so it now
takes the same recovery: `_parsed_store` grades the parsed document and raises
`ValueError` naming the fault, the bytes are kept aside, the reason is recorded
for `store_corruption()`, a working default takes over, and enrollment stays
sealed to the machine while it does. A store missing only optional keys, or
carrying keys this build does not know, is still served exactly as written - a
repair there would destroy live passkeys.
