### Fixed: `eval_policy` / `evaluate_benchmark` feed an open recording

Run under `start_recording`, an evaluation advanced every episode and wrote no
frame unless the caller passed an `on_frame` that called `add_frame` - which an
agent cannot do - so the run reported a success rate and the dataset could not
be closed. Both facades now install the backend's recording hook (the recording
half of the rollout hook, factored out of `_make_run_policy_hook` on MuJoCo,
Isaac and Newton), write one dataset episode per evaluation episode, and say
what they recorded. A caller's own `on_frame` is kept untouched - chaining would
double-write a hook that already records - and a recorder it never fed is named
in the evaluation's own answer instead of only at `stop_recording`.


The frames are labelled with the instruction the policy was conditioned on - the
caller's, else the benchmark's own `spec.instruction` (#187), read through one
shared `benchmark.spec_instruction` helper. A recorded `evaluate_benchmark` wrote
its dataset's `task` column as `"untitled"` for a rollout the policy had been
told to "pick up the red cube and lift it".
