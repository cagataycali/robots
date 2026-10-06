### Fixed: a `run_policy` that moved nothing reports "Policy failed", not "Policy complete"

A rollout shorter than the three-step fail-fast window, whose every step resolved
no actuator or commanded none, returned `status="error"` with a report whose first
line read "Policy complete on '<robot>'". The header now reads "Policy failed", the
same words the in-window fail-fast already used, so the status and the first line agree.
