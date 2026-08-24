### Quality: drive `start_recording`'s caller-input refusals on the Newton and Isaac backends

`start_recording` refuses four caller inputs before it touches a dataset - the
`fps` domain, the `push_to_hub` / `overwrite` posture flags, the `cameras` name
list, and a rate an in-flight rollout is not capturing at. Every backend's copy
was proven to *call* each shared guard by an AST sweep, on the stated grounds
that the Isaac and Newton backends "need their simulators installed to drive".
Both backends already have skeleton harnesses that drive `start_recording`
end to end without those simulators, and the three caller-input guards run
above the lerobot-extra probe, so none of them needs the dataset stack either.

A structural pin proves the guard is *called*, never that its refusal is
*returned*: keeping the call and dropping the `return` satisfies it. Measured,
six of eight such mutations across the two backends were invisible to the 287
tests covering these contracts. `tests/simulation/test_recording_preflight_refusals_across_backends.py`
drives the three reachable refusals on both backends - byte-identical to the
shared domain's own verdict - and pins that each returns before any dataset
directory is created, before the recording flag is set and before the
lerobot-extra probe. The fourth refusal is shown to be *unreachable* there
rather than untested: the rate guard reads `_active_rollout_rates`, which only
the MuJoCo backend overrides. The four sweeps' docstrings now record which half
each one holds.
