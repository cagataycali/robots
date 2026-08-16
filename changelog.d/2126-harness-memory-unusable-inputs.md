### Quality: pin every unusable-input branch in the harness-memory tool, and document the two raises that had no `Raises:` entry

`strands_robots/tools/harness_memory.py` has five branches that report an
unusable input or store, and together they were the module's entire uncovered
set. Four refuse -- the simulation tool spec carries no `action` enum, a trace
entry or a summary cannot be serialized, a global-rule store is not valid
UTF-8 -- and one degrades, recording an unknown library version in trace
provenance when the distribution metadata is absent rather than failing the
save. Each is a documented contract that nothing exercised, so a regression in
any of them was invisible: six plausible ones are caught by the new tests and
none by the 71 cases the module's tests already ran.

Two of them reach a caller through `HarnessMemory` rather than through the
tool, and neither documented that it can raise. `load_rules` and `append_rule`
both propagate the non-UTF-8 `ValueError`, and `load_rules` does so
all-or-nothing: rules are loaded together into one prompt, so returning a
partial mapping would present a store that could not be read as a kind with no
rules. Both now carry a `Raises:` entry stating that, and the behaviour it
describes is pinned. No behaviour changes -- the refusals and the fallback are
the ones that already shipped.
