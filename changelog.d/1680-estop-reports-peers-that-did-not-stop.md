### Fixed: an emergency stop no longer reports a peer as halted when nothing was stopped

`Mesh._dispatch` answered `{"ok": True}` for `action="stop"` when the registered
robot exposed no `stop_task`. Nothing was stopped, yet the peer returned an
affirmative acknowledgement:

```python
Mesh(status_only_robot, peer_id="arm-2")._dispatch({"action": "stop"})
# before: {"ok": True}      <- nothing was stopped
# after:  {"ok": False, "error": "peer exposes no stop_task; nothing was stopped"}
```

`emergency_stop` then folded that reply into `responses_received`, so an operator
watching a fleet E-STOP read a clean acknowledgement from a robot that was still
executing - on a safety path an affirmative lie is the worst available failure
mode, because it is the one that stops the operator from reaching for the
hardware cutoff.

The dispatch now reports the failure and logs it at ERROR, and `emergency_stop`
accounts for such peers separately: logged at CRITICAL and carried in both the
`strands/safety/estop` envelope and the audit record as `peers_not_stopped`.
`responses_received` keeps its original meaning (replies received), so the two
numbers can be compared rather than one silently absorbing the other.

The accounting is deliberately conservative - only responses that
*affirmatively* report failure (`ok is False`, or `status == "error"` from a
`stop_task` that itself failed) are flagged. An unrecognised response shape is
left out rather than guessed at, because a false "did not stop" on the safety
path trains operators to ignore the warning. Peers that never answered at all
remain visible as the gap between `responses_received` and the known peer count.
The local lockout still engages regardless of what any peer reports.
