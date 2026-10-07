### Tests: the rollout seed and recording posture refusals are wired once per surface (201 cells -> 100)

`tests/simulation/test_rollout_seed_is_applied_or_refused.py` built a MuJoCo
world for every unusable seed at every rollout surface (eleven values times
six surfaces, plus nineteen more for the shared-domain check), and
`tests/simulation/test_recording_posture_flag_domain.py` recorded a dataset
episode for each of seven truthy non-booleans. Which values are refused is a
property of the shared guard, so the whole value table is now pinned on the
guard with no world; each surface drives one value per way it can fail (a
type, and the per-caller bound) to prove it returns that refusal before it
acts. Same covered lines; 201 cells became 100.
