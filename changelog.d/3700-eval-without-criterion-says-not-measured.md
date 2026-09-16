### Fixed: eval_policy without a success criterion prints no success fraction

`eval_policy` / `evaluate` with no `success_fn` and no benchmark spec used to
report `Success: 0/3 (0.0%) [no success criterion - not measured]`. An agent
asked for a baseline read that as a 0% success rate and marked every episode
failed. The text now says `Success: not measured (no success criterion - pass
success_fn, e.g. 'contact', or a benchmark spec)` and prints no fraction. The
json payload is unchanged (`success_rate: 0.0`, `success_measured: false`).
