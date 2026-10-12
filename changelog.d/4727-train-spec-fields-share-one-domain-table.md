### Fixed: an array-valued posture flag is reported by SageMaker and Isaac Lab instead of raising

`SagemakerTrainer.validate` and `IsaacLabTrainer.validate` raised `ValueError`
("the truth value of an array ... is ambiguous") when `resume` or `streaming`
was a NumPy array: the SageMaker hyperparameter mapping compared the value with
`""`, and Isaac Lab read `spec.resume` by truthiness after its own gate had
already refused it. Both now report the flag through the shared posture-flag
refusal, like every other backend. The supervised `TrainSpec` field suites -
learning rate, seed, run size, checkpoint cadence, validation episodes, launch
topology, posture flags and LoRA shape - are one reader table in
`tests/training/test_train_spec_field_domains.py`, which is what found the two.
