### Fixed: a seed the client reseed cannot apply no longer half-seeds the process

`reseed_client_rngs` is the applier `Gr00tPolicy.reset` and `Cosmos3Policy.reset`
both route through, so it is how a rollout makes an episode reproducible. It
reseeds Python `random`, then NumPy's legacy global RNG, then torch - three
appliers with three different accepted domains, in that order, inside one
`try` whose `except Exception` logs at `INFO` and swallows.

Only the second of them bounds the value. So for every seed NumPy refuses -
anything negative, fractional, non-integral, or above `MAX_EVAL_SEED` - Python
`random` was reseeded, NumPy was not, and `reset` returned normally. Seven of
twelve probe values left that state:

| seed | Python `random` | NumPy | `reset` |
| --- | --- | --- | --- |
| `7`, `0` | reseeded | reseeded | returns |
| `None` | untouched | untouched | returns (documented no-op) |
| `-1`, `2.5`, `3.0`, `"abc"`, `nan`, `2**32`, `MAX_EVAL_SEED + 1` | **reseeded** | **untouched** | **returns** |
| `True` | reseeded (seed 1) | reseeded (seed 1) | returns |

A caller was therefore told the episode was seeded while half the streams a
stochastic policy draws from were not, with the reason visible only at `INFO`.
That is worse than a refusal: nothing distinguishes it from success, and
partial reproducibility is not a weaker form of reproducibility.

The seed is now checked before the first applier runs, so the reseed is
all-or-nothing for every value it accepts. The domain is the one the sibling
applier `set_eval_seed` already enforces, down to the `MAX_EVAL_SEED` ceiling
that `randomization_seed_error` carries for exactly this destination - both
reseed the same legacy NumPy global RNG, so both can honor the same seeds.
`set_eval_seed`'s own docstring already called that ceiling "the ceiling every
rollout surface accepts"; this is the rollout surface that was not applying it.

Refusing rather than logging is what lets a caller tell "this episode is
reproducible" from "it is not". The module's best-effort clause is unchanged
and still covers what it was written for - an applier that *fails*, such as an
absent torch - because swallowing that leaves every RNG consistently unseeded,
which is a different outcome from leaving half of them seeded.

Pinned by `tests/policies/test_client_reseed_seed_domain.py`: a refused seed
moves neither RNG, an accepted seed reaches all three appliers, and the two
appliers' verdicts are asserted equal over the whole probe set so neither can
drift from the other.
