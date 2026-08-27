### Fixed: every WBC observation dimension is held to the shared count domain

`WBCConfig` carries five discrete dimensions - `num_actions`, `obs_history_len`,
`single_obs_dim`, `command_dim` and `n_obs_joints` - and each is consumed as a
`deque` maxlen, a `range()` bound, a slice index or an `np.zeros` width. Every
scalar in the same `__post_init__` already went through a shared numeric domain,
under a comment explaining why: each is "read verbatim into either the PD law
that writes `data.ctrl` or the observation the network sees", so a non-real value
"surfaces as a bare `TypeError` from its `float()` after the ONNX sessions have
loaded and the rollout has started - the mid-rollout failure this module exists
to convert into a construction-time message". The dimensions above them were
checked with bare comparisons (`< 1`, and `< 3` for the command block), which
decide a floor and cannot decide whether the value is an integer at all.

Two spellings got through, and `positive_count_error` - the shared domain for
exactly this kind of value - names the first of them in its own docstring:
"`bool` is rejected explicitly. It is an `int` subclass, so a bare `value < 1`
test lets `True` through as a silent count of 1."

Measured against the shipped 86-wide GEAR-SONIC layout stacked over six frames:

| `obs_history_len` | before | after |
| --- | --- | --- |
| `6` (default) | 6 frames, 516-wide input | unchanged |
| `True` | **1 frame, 86-wide input, no error** | refused, names the field |
| `nan` | accepted; bare numpy `TypeError` at the first frame | refused, names the field |
| `2.5` | accepted; bare `TypeError: an integer is required` | refused, names the field |

The `True` row is the one that produced no error at all. `ObservationHistory`
took `maxlen=1`, `num_obs` became `86 * True == 86`, and `push` returned an
86-wide vector for a checkpoint expecting 516 - a stacked observation five frames
short, reported as healthy. `nan` is below nothing, so it passed all five
comparisons and reached the observation builder, which is the mid-rollout failure
the value checks below it exist to prevent. Across the five fields and thirteen
spellings, sixty cells were accepted or raised an unnamed error; all sixty are
now refused at construction with the field named.

The three `>= 1` floors are exactly what the shared domain decides, so they are
gone rather than restated. The command block's floor of three and the
`n_obs_joints >= num_actions` relation are not, and survive with their own
messages, asked of values the domain accepted - so `command_dim=2` still reports
"must be >= 3 (vx, vy, omega)" while `command_dim=True` now reports that a flag
is not a count, rather than a width slightly too small.

The domain is strict-`int` rather than any-integral-real because these values
reach a C-level API: `deque(maxlen=np.int64(6))` and `np.zeros(86.5)` both raise,
so a domain accepting an integral float or a NumPy integer would hand the caller
a value the next call cannot use. Not changed here: that domain accepts an
arbitrarily large `int`, which `deque` then refuses with an `OverflowError` naming
nothing - a property of a guard with fifty-one callers rather than of this config.
