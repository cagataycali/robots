### Fixed: FastSAC refuses an entropy-temperature learning rate its optimizer cannot be driven by

`FastSacTrainer` builds two optimizers from two separate learning-rate fields --
`learning_rate` for the actor and both critics, and `alpha_lr` for the entropy
temperature -- and passes each straight to `torch.optim.Adam(..., lr=...)`. Only
the first went through the shared optimization preflight, so every failure mode
that gate documents stayed reachable through the second field, which
`RLTrainSpec` documents as the "Learning rate for the temperature optimizer
(SAC)".

Measured on a 40-timestep run starting from `init_alpha=1.0`. `alpha_lr=0` (and
`0.0`, and `False`) builds the optimizer and moves the temperature by nothing, so
`autotune_alpha=True` silently behaves exactly like `autotune_alpha=False` while
reporting the automatic temperature it was asked for. `inf` also builds, and
sends the temperature to an infinity on the first step; because the temperature
multiplies the log-probability in the *actor* loss the damage is not confined to
it, and the run finished `status="success"` with a checkpoint whose largest
parameter magnitude was `inf`. `True` was a silent rate of `1.0`, over three
thousand times the `3e-4` default. A negative value and `nan` are refused by
`torch.optim.Adam`, and a `str` / `None` / `list` raises a bare `TypeError`
naming neither the field nor the value -- but all four only in `setup`, after the
env and both networks are built, past the point `validate()` documents itself as
running before.

`validate()` now checks the field against the same
`positive_finite_number_error` domain as the first rate, via a new
`temperature_learning_rate_problems` gate reached through
`Trainer._temperature_learning_rate_problems`. Both rates are reported in the
same preflight rather than one round at a time. Unlike `learning_rate` there is
no `None` sentinel to exempt: `alpha_lr` is annotated `float` with a concrete
`3e-4` default, so `None` is a value the optimizer cannot take. The check is
inert unless `autotune_alpha` is set, which is the only branch that constructs a
temperature optimizer, and scoped to the off-policy backend -- PPO and the mock
trainer never read the field and must not report on it. A structural test derives
the set of modules in scope from the tree, so a second backend that starts
driving a temperature optimizer with the field fails until it routes through the
gate.

`init_alpha` and `target_entropy`, the two neighbouring fields of the same
temperature block, are out of scope here: `init_alpha=0` yields a temperature of
zero, which is a coherent "no entropy bonus" configuration rather than an
unusable value, and `target_entropy` is signed by construction (it defaults to
`-num_actions`), so both need a different domain and a decision this change does
not make.
