### Docs: the development setup installs the `hatch` task runner

The README, contributing guide and AGENTS.md told a new contributor to run `hatch run test` right after `uv pip install -e ".[all,dev]"`, but `hatch` is in no extra (the build system names `hatchling`, the backend, not the CLI), so the second command failed with `hatch: command not found`. The install line now names `hatch`, matching what CI already installs before `hatch run lint`.
