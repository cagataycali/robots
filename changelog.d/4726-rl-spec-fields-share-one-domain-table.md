### Tests: the RL hyperparameter refusals are one table

Thirteen RL domain suites (`gamma`, `lam`, `num_learning_epochs`,
`max_grad_norm`, `clip_param`, the two loss weights, `alpha_lr`, `init_alpha`,
`target_entropy`, `policy_delay`, the three TD3 noise scales, `tau` and the
three replay counts) each re-asked the same questions through `validate`: is
it refused, in the shared words, only by the backends that read it, and does
`train` stop before building anything. Those questions are now one table in
`tests/training/test_rl_spec_field_domains.py`; each suite keeps what its field
does to the run that consumes it. 326 cells become 132, and every planted guard
removal still fails.
