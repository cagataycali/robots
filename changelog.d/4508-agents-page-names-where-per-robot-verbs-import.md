### Docs: the agents page names the package the G1 and Reachy Mini verbs import from

`docs/learn/agents.md` said `strands_robots.tools` lazy-loads every tool, but
`use_unitree`, `g1_*` and `reachy_*` import from `strands_robots.tools.g1` and
`strands_robots.tools.reachy`. Those rows now name their package, and a test
imports every tool in the table from the module its row names.
