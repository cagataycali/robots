### Quality: pin the `policy_kwargs` refusal `start_policy` never returned

`_validate_policy_mapping` guards two opaque keyword bags -- `policy_config`
(splatted by `create_policy`) and `policy_kwargs` (splatted by
`Policy.get_actions`) -- across four rollout entry points. Seven of those eight
`(surface, parameter)` refusals were returned by some case in the suite. The
eighth, `start_policy`'s `policy_kwargs`, never was, and it is the cell the
guard exists for: both splats happen on the worker thread, so without the
pre-flight the caller is told a policy started for a rollout that never produces
an action. Unlike the recording-rate guard on the same method there is no
structural sweep over this one either, so deleting the guard outright was also
invisible to the module's other sixteen cases.

One behavioural case now pins it: the verdict, the envelope as the shared
`policy_mapping_error()` message verbatim, that no worker was submitted and the
robot is not marked running, and that the rejected call never consumed the
per-robot slot. Tests only; no library behaviour changes.
