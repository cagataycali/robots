### Quality: the rollout step horizon is pinned once, not three times

Two test modules graded the same `n_steps` / `max_steps` contract: one swept
every control frequency from 1 to 120 Hz to prove a one-step rollout runs one
step, and the other already pinned the frequencies where the old float round
trip lost a step, plus a structural check over every rollout loop in the
package. Each refusal value was also run five times to read five facets of one
message. The duplicates are gone and each refusal is one cell. The two modules
ran 413 tests in 94 worker-seconds before and 192 in 60 after, with the same
covered lines.
