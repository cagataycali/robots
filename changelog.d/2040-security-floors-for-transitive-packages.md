### Security: declare the advisory floors that keep four transitive packages patched

`cbor2`, `ujson`, `twisted` and `pyopenssl` reach the dependency graph only
through another dependency -- cbor2/ujson under `autobahn`, twisted under
`roslibpy`, pyopenssl under twisted -- so nothing in `[project]` declared a
version for any of them. All four resolved above a HIGH advisory, which is why
`dependency-review-action` was green, but that was the resolver's choice rather
than a stated requirement, and `cbor2` sat exactly on its patch floor (5.9.0 is
the first patched version for GHSA-3c37-wwvx-h642) with no margin at all. Any
input that moved one of them down would have re-introduced a HIGH advisory, and
the failure would have surfaced as `dependency-review-action` going red on a
pull request that touched something unrelated.

Each floor is now declared in `[tool.uv] constraint-dependencies` at the first
version clearing every HIGH/CRITICAL advisory for that package, with the GHSA id
named beside it. A constraint rather than an override: measured on this manifest,
`gymnasium>=1.1.1` as a constraint fails the resolution and names the
`[vera-sim]` extra's contradicting `gymnasium==0.29.1`, while the same floor as
an override resolves silently and discards that requirement -- so an override
would hide the one thing worth knowing. Declaring the floors changed no resolved
version.
