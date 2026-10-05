### Fixed

- `docs/learn/agents.md` no longer says `strands_robots.tools` lazy-loads every tool: the Unitree G1 and Reachy Mini rows name the sub-package their verbs import from (`strands_robots.tools.g1`, `strands_robots.tools.reachy`), and a test imports every tool in that table from the module its row names.
