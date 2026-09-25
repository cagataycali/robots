### Docs: the rollouts page keeps the call, what it reports gets its own page

`docs/simulation/rollouts.md` documented two subjects in one 1,966-word scroll:
driving a rollout (the action table, the parameter domains and the posture-flag
refusals, stopping and what counts as running) and reading one back (the json
payload field by field, the action-health fields that expose a
crippled-but-successful run, `seed`, the per-episode video config, and the
async-RTC chunk pipeline with its prefetch telemetry). The second half moves to
`docs/simulation/rollout-results.md`, leaving 943 and 1,142 words, so neither
page owes the 1,500-word budget an exemption. No prose was rewritten - the
boundary is the page's own H2, so the moved block is verbatim, and the two
sections that reference each other ("see below", "the telemetry below") travel
together. The call page keeps a pointer in its intro and a see-also row, and the
new page has a nav row.
