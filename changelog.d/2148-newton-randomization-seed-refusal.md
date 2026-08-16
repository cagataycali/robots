### Quality: drive the seed refusal on Newton's randomization entry points

`randomize` and `set_obs_noise` store a seed and draw from it later - inside
`_rebuild` and on the first observation drawn - so both refuse a seed
`numpy.random.default_rng` cannot take at the call that supplied it. On the
Newton backend that refusal was pinned structurally only: the cross-backend
parity check asserts by AST that every backend *calls* the shared
`randomization_seed_error`, which a call whose verdict is discarded still
satisfies, and the behavioural pin beside it drives MuJoCo. Both refusal
branches were unreached, while the sibling range and amplitude guards on the
same two methods were driven directly.

Adds the behavioural half on Newton, on the pure-Python host the sibling guard
tests already use, so it needs neither Newton nor Warp: both entry points return
the shared domain's verdict verbatim, a refused call configures nothing and
never rebuilds, usable seeds are still applied, and this path keeps
`default_rng`'s full integer width rather than the narrower ceiling a rollout
seed carries. `strands_robots/simulation/newton/randomization.py` goes from 98%
to 100% line coverage. Tests and docstrings only.
