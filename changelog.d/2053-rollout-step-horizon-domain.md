### Fixed: refuse a rollout step horizon that is not a whole number of steps

`run_policy` / `start_policy` / `run_multi_policy` resolved their step-count
horizon (`n_steps`, or the legacy `max_steps` alias) with a bare `<= 0` test,
which only sees the sign. `n_steps=2.7` ran two steps and `n_steps=True` ran
one, each reported as a successful rollout of a horizon the caller never asked
for, while every sibling count on the same call - `n_episodes`,
`action_horizon`, `control_substeps` - and the identically-named `eval_policy`
step budget already refused both. The horizon now goes through the same shared
positive-count domain, so one parameter name no longer carries two contracts.

A non-positive `max_steps` also reported `n_steps must be > 0`, naming a
parameter the caller never passed, because the alias was normalized before the
check; it is now validated under its own name.

`run_policy`'s `n_steps`, `max_steps`, `policy_object` and
`max_onframe_failures` had no `Args:` entry, so the domain enforced above was
undiscoverable from the API docs - the `duration` entry told a reader that a
step count "wins" over it without either count being documented anywhere. All
four are now documented, and a test pins every `run_policy` parameter as
documented so a future one cannot slip in undocumented the same way.
