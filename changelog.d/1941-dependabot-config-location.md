### Fixed: the Dependabot config is read from `.github/` instead of parsed as a workflow

`dependabot.yml` is a configuration file, not a workflow, and it was in
`.github/workflows/`. Actions therefore parsed it as a workflow and rejected it
at parse time on every push to `main` (15 of 15 sampled pushes, `total_jobs: 0`),
while Dependabot — which reads version-update config from `.github/dependabot.yml`
only — never saw it at all, so both ecosystem entries and the `ml-stack` /
`sim-stack` / `dev-tools` grouping were inert.

The silent half is the one with consequences: AGENTS.md > "Action Pinning"
delegates action SHA-pin freshness to the `github-actions` ecosystem entry and
calls that pin non-negotiable for `pypa/gh-action-pypi-publish`, which tracks a
moving `release/v1` branch. That control was not running.

The move is a pure rename; no configuration content changed.
`tests/test_dependabot_config_location.py` pins the location, both ecosystem
entries and AGENTS.md's delegation, and guards the class by asserting every file
under `.github/workflows/` declares the `on:` and `jobs:` keys Actions requires.
