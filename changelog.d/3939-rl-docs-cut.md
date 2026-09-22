### Changed: the reinforcement-learning docs are a guide and an API reference, both within the word budget

`docs/training/rl.md` had grown to 4,868 words by documenting its own review
history: every validated `RLTrainSpec` field carried a paragraph or two on what
an unusable value used to do before `validate()` graded it, down to the measured
parameter sums and `log_alpha` readings of the runs that found each one. That is
CHANGELOG material, and it buried the contract a caller actually needs -- the
field, its domain, and which backend reads it.

Each field is now one row of a table, and the page splits at its H2s: `rl.md`
(984 words) trains and deploys a checkpoint, and the new
`docs/training/rl-reference.md` (1,367 words) carries the `SimEnv` argument
domains, the `BaseRLAlgo` lifecycle and `evaluate()`, every `RLTrainSpec` field,
and the device rule. No documented fact was dropped: every code identifier the
old page named survives on one of the two pages. Both are under the 1,500-word
ceiling, so `training/rl.md` leaves the `_OVER_BUDGET` roster and the site falls
from 115,724 to 113,207 words.
