### Fixed

- A native driver's `start_task` now runs the provider's own `Policy.preflight`
  before it builds the policy, so a configuration the provider refuses without
  constructing is refused by the verb instead of starting a rollout on an
  energized arm that faults at step 0.
