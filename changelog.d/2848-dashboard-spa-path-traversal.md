### Fixed: the dashboard SPA catch-all refuses a URL that walks out of `frontend/dist`

The `GET /{path:path}` handler introduced with the dashboard forwarded the URL
path straight into `FRONTEND_DIST / path`, then served the result with
`FileResponse` if `.is_file()` was true. That trusts the client to describe a
file inside `dist`, and the trust is not warranted: `GET /%2E%2E/secret.txt`
against a stock dashboard resolves to the sibling of `frontend/dist` and
streams its bytes back with a `200 OK`. Measured against the pre-fix tree, four
URL shapes all reach a `dist`-sibling file on disk - two literal (`/../` and
`/foo/../../`) and two URL-encoded (`/%2E%2E/`, `/..%2F`); the encoded pair is
what an attacker actually reaches for, because the httpx client folds the
literal `..` before it reaches the handler while browsers and proxies do not.
CodeQL's `py/path-injection` reported exactly this data flow (untrusted URL
path -> `pathlib.Path` argument -> `FileResponse`) on PR #2848; grading it as
a live path traversal rather than a style alert is the intent of this change.

The fix resolves the frontend root once at mount time (so a symlink swap
between the check and the send cannot race), then, for every request, resolves
the candidate and asks it whether it is a descendant of that root via
`Path.relative_to`. A candidate that does not answer is refused, and the SPA
catch-all falls back to `index.html` - the same behaviour a legitimate SPA
route receives, so nothing legitimate breaks. `tests/test_dashboard_spa_path_
traversal.py` grades the confinement by driving the same URL shapes through
`TestClient` (with a real dist under `tmp_path` and a real secret file next to
it), so a rewrite that keeps the resolve step but drops the descendant check
still fails. Two independent evidences per hostile URL: the response body
must not carry the secret, and the sibling-file lookalike (`../sibling/app.js`)
must not be served as the legitimate one.
