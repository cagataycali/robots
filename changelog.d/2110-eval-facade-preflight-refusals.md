### Quality: pin the pre-flight refusals the two eval facades never returned

`run_policy` / `eval_policy` / `evaluate_benchmark` each open with a block of
pre-flight guards so a bad option "costs no weight download and no frame". The
reference facade pins all twelve of its refusals; six on the two eval facades had
never been returned by any test - `evaluate_benchmark`'s `video`,
`policy_config`, `policy_kwargs` and dataset-rate checks, plus `eval_policy`'s
`policy_kwargs` and dataset-rate checks. What was verified was that the guard is
wired, not that it refuses: deleting four of the six changed nothing in the suite,
and keeping the rate guard's call while discarding its refusal passed the AST
parity test that covers it. Each refusal is now pinned as the shared rule's
verdict verbatim, together with the guard-order property the block exists for -
no benchmark lookup, no policy build, no frame and no motion.
