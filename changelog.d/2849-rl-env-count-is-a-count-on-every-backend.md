### Fixed: `num_envs` is held to the shared count domain on both RL backends

`RLTrainSpec.num_envs` is the third caller-supplied factor of the one loop bound
every RL backend derives, `max(1, total_timesteps // (rollout_steps * num_envs))`
iterated as `range()`. Its two siblings are held to
`strands_robots.utils.positive_count_error` by `rl_run_size_problems`, whose own
docstring works through why a bare `value <= 0` test on a *derived* bound is
weaker than the same test on a field read straight off the spec: the `max(1, ...)`
clamp turns every value that survives the comparison but cannot divide into a
silent wrong-length run.

`num_envs` was deliberately left out of that gate, for a reason that holds: which
*counts* are usable genuinely differs per backend - PPO parallelizes over any
positive count where the MuJoCo-backed FastSAC is single-env and requires exactly
`1` - so that half is not one shared rule, and each backend stated its own with a
bare comparison. What the reason does not cover is whether the value is a count
at all. That half is identical on both backends and was stated by neither, so the
third factor of this product sat outside the domain its two siblings are held to.

Measured over `total_timesteps=1000` and `rollout_steps=64`, which asks for 15
iterations of 64 steps:

- `nan` passed PPO's `< 1`. `64 * nan` is `nan`, `max(1, nan)` keeps the `1`, and
  `1000 // 1` is `1000`: the run reported `success` and announced `"1000
  iterations x 1 steps complete"` while each of those 1000 iterations collected
  `rollout_steps` steps - 66x the requested budget, under a message that
  misdescribes both factors.
- `inf` passed the same test and ran **one** iteration, announcing `"1 iterations
  x inf steps complete"`.
- `1.0` and `2.5` kept the bound a float, which raises `TypeError: 'float' object
  cannot be interpreted as an integer` out of `range()` - after `setup` had built
  the environment, the networks and the optimizers, which is the cost a read-only
  preflight exists to precede. `1.0` reached that through FastSAC's `!= 1` as
  well. A large float does not: `100.5` makes `1000 // 6432.0` the float `0.0`,
  which the clamp replaces with the int `1`, so it runs one iteration instead.
- `True` passed both backends - it satisfies `< 1` being false and `!= 1` being
  false - and is numerically `1`, so the run length was right and what was wrong
  is that a value reading as a flag was accepted as a count. That is the reason
  the shared domain refuses a `bool` rather than testing a bound.
- `"4"` and `None` raised `TypeError: '<' not supported between instances of
  'str' and 'int'` out of PPO's comparison itself, from a `Trainer.validate`
  documented to *return* problems.

Both backends now consult the shared domain first and ask their own count rule
only of a count, matching the idiom each already uses one statement later for a
relation between two counts. The per-backend line is unchanged and pinned: `4` is
still usable for the parallel backend and still refused as single-env by the
other, and a non-count is no longer reported as the wrong *number* of
environments. PPO keeps no second comparison, because over the values the domain
accepts every positive `int` parallelizes, so a `< 1` branch there would be
unreachable - and the refusal now names the backend that refused, which the bare
comparison's message did not.

The scope paragraph in `rl_run_size_problems` said the whole field was excluded;
it now says which half, so the next reader is not invited to leave the shared half
open again. `TestNumEnvsIsNotInTheSharedDomain`'s non-vacuity test asserted that
"excluding it from the domain did not leave it unchecked" while driving only `0` -
the one unusable value a bare comparison already caught - and is widened to the
values that survive one.
