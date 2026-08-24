### Fixed: a rollout seed is applied or refused, not silently dropped

`seed` reached exactly one statement in `PolicyRunner.evaluate` - the
`_evaluate_with_spec` delegation - so the plain `success_fn` loop below it never
read the value. Every `SimEngine.eval_policy` call lands in that loop, because
that facade exposes no `spec` parameter, so the whole surface accepted a
reproducibility seed and discarded it: two evals at `seed=7` drew different
actions, `policy.reset(seed=...)` was never forwarded, and both reported
`status="success"`. `eval_policy`'s own docstring advertises that it and
`run_policy` behave the same way, and `run()`'s comment describes itself as
mirroring "the per-episode reseed in `evaluate()`" that did not exist there.

The seed is now applied on that path exactly as on the spec path: reseeded once
from the master seed, then per episode from a master RNG derived from it, with
each per-episode seed forwarded to `policy.reset` so a service-mode policy can
reseed its own process. A `None` seed leaves RNG state untouched, so an unseeded
eval acquires no global side effect it did not have.

Honoring the seed also settles its domain, because an unusable value cannot be
applied at all - the seed ends at `numpy.random.seed` / `default_rng`. That half
had been going three ways, none of them naming the parameter: `run_policy` raised
NumPy's own `TypeError: Cannot cast scalar from dtype('float64') to
dtype('int64')` out of a method documented to return a structured result and
bound as an agent-tool action; `start_policy` reported "started" and failed on
its worker thread; and `True` was accepted everywhere as a silent seed of `1`.
`run_policy` / `eval_policy` / `start_policy` / `evaluate_benchmark`,
`PolicyRunner.run` / `.evaluate` and the `run_policy` agent tool now share
`randomization_seed_error` - the same domain `randomize` and `set_obs_noise`
already used - so a seed refused for one cannot be accepted for the rollout whose
reproducibility it is supposed to pin. The tool refuses it before it opens a
dataset, matching its sibling pre-flight checks.

The rollout surfaces carry one bound the randomization surfaces do not, because
their appliers are not equally wide. `set_eval_seed` reseeds the legacy NumPy
global RNG (`numpy.random.seed`) as well as `default_rng`, and that one refuses
anything above `2**32 - 1` - so a seed in `[2**32, inf)`, which includes the
common millisecond-epoch timestamp idiom, was accepted and then raised NumPy's
own `Seed must be between 0 and 2**32 - 1` from inside the rollout: past the
envelope on `eval_policy`, and on a worker thread after `start_policy` had
already reported "started". `randomization_seed_error` now takes an optional
ceiling, and the rollout surfaces pass `MAX_EVAL_SEED`; `randomize` /
`set_obs_noise` reach only `default_rng` and keep the unbounded domain they can
honor. `set_eval_seed` enforces it too, since it is public API documented for
direct callers, and the `run_policy` tool checks the seed its *last* episode
derives (`seed + n_episodes - 1`) rather than only the one it was handed.

`MAX_EVAL_SEED` lives in `simulation/base.py`, beside the `max_seed` parameter it
feeds, rather than in `policy_runner.py` with the applier it describes. The note
above `base.py`'s `policy_runner` import records why: CodeQL's
`py/unsafe-cyclic-import` walks `TYPE_CHECKING` blocks, and `policy_runner`
imports `SimEngine` from `base` under one, so any name added to that module-level
import line closes an AST-visible cycle. The rollout side reaches the constant
through the function-local import it already uses for `randomization_seed_error`,
so neither module gains a module-level edge.
