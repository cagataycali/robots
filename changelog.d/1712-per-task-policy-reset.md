### Fixed: a hardware task starts its policy from a clean per-episode state

`Robot._execute_task_async` never called `Policy.reset`, so a policy object
driven through more than one task carried the previous task's per-episode
state. `Policy.reset` exists to clear exactly that state - action chunk
caches, sampler RNG, KV-caches - and the sim runner already calls it once per
episode; only the hardware loop, where the leftover actions are commanded to a
physical arm, skipped it. Driving one chunk-caching policy through two tasks
replayed task one's cached chunk under task two's instruction, so the arm
executed actions inferred for an instruction the caller never gave.

The control loop now resets the policy once per task, before the first action
reaches the bus. The reset is best-effort, matching the sim runner: a policy
whose `reset` raises (for example one forwarding to an unreachable inference
server) is still driven, and the warning names the stale state.
