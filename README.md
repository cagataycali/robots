# cleanup policy_stop_timeout budget

`capture.py` runs in a checkout and writes `facts.json` + `reference.png`: it starts a
live 50 Hz rollout, sleeps 0.5 s, calls `cleanup(policy_stop_timeout=X)` once per
candidate budget, and records the wait, whether the join completed before the world
was freed, whether the budget was reported, and any log record the `%.1f` dropped.
Run it once at `upstream/main` and once on the branch.

`compose.py` builds the figure from the two dumps. It asserts the dumps came from
different trees, that the reference rollout's joints are identical, and every count
the figure states.
