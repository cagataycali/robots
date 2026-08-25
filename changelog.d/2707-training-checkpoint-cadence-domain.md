### Fixed: the checkpoint cadence a TrainSpec asks for is one shared whole-number domain

`TrainSpec.save_freq` is the interval, in optimizer steps, at which a run writes
a checkpoint. Four providers read it - the three supervised backends and the
SageMaker transport - and it was the only numeric field among them with no
domain, while `steps`, the step count interpolated into the very same argv,
already shared one. `validate()` reported nothing about it on any backend for
any of `2.7`, `5000.0`, `True`, `False`, `nan`, `inf`, `"5000"` or `None`.

Each provider consumes the value in three shapes, and every way an unusable
cadence failed was silent or late:

* A `spec.save_freq > 0` selector, which LeRobot uses twice to derive the
  validation cadence (`--eval_steps=` on the argv path, `cfg.eval_steps` on the
  in-process one) because a non-positive cadence disables periodic saving. `nan`
  compares false against everything, so it took that *disabled* branch: a spec
  asking to checkpoint every `nan` steps built `--save_freq=nan
  --eval_steps=100` and evaluated once at the very end instead, under a
  successful result. `True` and `inf` are greater than zero, so they passed
  through as the cadence itself (`--eval_steps=True`, `--eval_steps=inf`).
* An argv or Hydra token - `--save_freq=` (LeRobot), `--save_steps=` (GR00T),
  `checkpoint.save_iter=` (Cosmos) - which interpolates the value verbatim, so
  every spelling rendered and failed, if at all, inside the launched run once the
  dataset and model were loaded. LeRobot declares the field as a plain `int`, and
  its own draccus decoder refuses `2.7`, `5000.0`, `True`, `nan` and `inf` with
  `DecodingError: Couldn't parse '2.7' into an int`.
* A direct assignment into a typed config - `cfg.save_freq`, GR00T's
  `save_steps` kwarg, a forwarded SageMaker hyperparameter - none of which
  coerces.

A string or `None` was worse than any of those: it raised `TypeError: '>' not
supported between instances of 'str' and 'int'` out of the comparison itself,
from inside a `Trainer.validate` documented to *return* problems.

`Trainer._checkpoint_cadence_problems` now routes the field through
`checkpoint_cadence_problems`, a fifteenth shared gate in
`strands_robots.training._validate`, and the message names the backend that
refused the value. The type test is strict for the same reason the run-size gate's
is: `save_freq` and `steps` reach the same `int` decoder in the same argv, so the
same number must not be refused for one step count and accepted for the next.

**The floor is deliberately not part of the domain.** LeRobot documents a
non-positive `save_freq` as disabling periodic saving - its
`should_save_checkpoint` implements exactly that, avoiding a `ZeroDivisionError`
from `step % 0` - and the `eval_steps` fallback above exists for that case, so
`0` and a negative remain first-class and the refusal names that spelling. A
cadence above `steps` is a legitimate configuration too. Neither endpoint is
decided here, and both are pinned so they cannot move silently.

The gate is scoped like the run-size one rather than made universal: `TrainSpec`
documents that a backend "reads the fields it supports and ignores the rest", and
the RL trainers never read the field, so they must not report on it. That scoping
is enforced structurally - the set of providers required to route through the gate
is derived from which modules actually read `save_freq`, by attribute access *or*
through a forwarding table, so a fifth backend that starts checkpointing from it
fails the parity test until it does.
