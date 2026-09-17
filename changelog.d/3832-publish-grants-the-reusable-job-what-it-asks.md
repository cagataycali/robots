### Fixed: the PyPI publish grants the reusable test job what it asks for, and can be re-run against a tag

Publishing `v0.5.2` never started: `pypi-publish-on-release.yml` calls
`test-lint.yml`, which since #3822 asks for `pull-requests: read`, and a
reusable workflow may hold no permission its caller did not grant - GitHub
refused the run before any job existed (`startup_failure`), so the tag exists
and PyPI still serves 0.5.1. The caller now grants it. The workflow also gains a
`workflow_dispatch` with a `tag` input, checked out in the build job so
hatch-vcs derives the version from that tag, so a refused publish is re-run
against the same release instead of re-publishing it. A grader now compares every caller's
`permissions` block against the scopes the workflow it calls asks for, so the
next one is a failing test rather than a refused run with no job to read.
