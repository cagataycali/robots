### Fixed: publishing a release runs the publish workflow again

`pypi-publish-on-release.yml` calls `test-lint.yml` before it builds, and #3822
gave that called job a `pull-requests: read` scope for its guards step. A called
workflow may not hold a scope its caller did not grant, and GitHub reports the
mismatch as a startup failure with no job to read. The `v0.5.2` release fired
the workflow and nothing ran: the tag reached GitHub, the wheel never reached
PyPI. The caller now grants the scope, as `ci.yml` already does, and every call site
in the tree is graded against the scopes its callee declares so that the next
one cannot be missed.
