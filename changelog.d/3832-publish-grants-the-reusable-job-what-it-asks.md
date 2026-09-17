### Fixed: the PyPI publish grants the reusable test job what it asks for, and can be re-run against a tag

Publishing `v0.5.2` never started: `pypi-publish-on-release.yml` calls
`test-lint.yml`, which since #3822 asks for `pull-requests: read`, and a
reusable workflow may hold no permission its caller did not grant - GitHub
refused the run before any job existed (`startup_failure`), so the tag exists
and PyPI still serves 0.5.1. The caller now grants it. Because a `release` event
runs the workflow file the tag names, repairing that file could not re-publish a
release already out, so the workflow also accepts a `workflow_dispatch` carrying
the tag: the reusable call and the build job's checkout both read it, and
hatch-vcs derives the released version from it instead of a branch.
