### Fixed: the optimizer learning rate is one shared positive-finite domain across every training backend

`TrainSpec.learning_rate` reaches every backend - the supervised three assign it
to their config's optimizer field (LeRobot `policy.optimizer_lr`, GR00T
`FinetuneConfig.learning_rate`, Cosmos `optimizer.lr`) and the RL trainers pass
it straight to `torch.optim.Adam` - and no backend checked it, while each bounded
its neighbours. `FastSacTrainer.validate` compares eight sibling numerics against
literals and skipped the one that decides whether any of that work updates a
weight; the supervised backends gate the two run-size factors and skipped it too.

Both ends of the domain failed silently rather than loudly. `learning_rate=0`
(and `False`) ran the full `steps` x `global_batch_size` of work and updated no
weight, so the run reported success and wrote a checkpoint identical to its
initialisation - the pathology the run-size gate exists to prevent, reached by a
different route and at full cost. `inf` diverged on the first step and wrote a
checkpoint of `NaN`, also under a successful result. `True` was honored as a
silent learning rate of `1.0`. A negative or `nan` rate *was* refused, but by
`torch.optim.Adam` only once the dataset and model were loaded, and a string or
list reached the config as a field of the wrong type.

`Trainer.validate` now reports an unusable rate through one shared
`learning_rate_problems` gate, against the same
`positive_finite_number_error` domain the rest of the library uses for a
continuous knob, with a message naming the backend that refused it. `None`
remains the documented "use the backend's own default" sentinel and every usable
rate is untouched. An AST guard asserts that every concrete `Trainer` - including
the two reached through `BaseRLAlgo` - routes through the shared gate rather than
re-implementing the rule.
