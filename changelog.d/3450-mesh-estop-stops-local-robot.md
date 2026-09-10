### Fixed:
- `Mesh.emergency_stop()` now stops the robot registered in the issuing process too. `broadcast` never reaches the sender (its own envelopes are dropped on receipt), so the one robot next to the operator was the one an e-stop never halted; its answer is now the first entry in the returned responses and counts in `peers_not_stopped`.
- `stop` is admitted while the emergency-stop lockout is engaged (alongside `status` and `resume`); a second e-stop reaching an already locked-out peer halts it instead of being rejected.
