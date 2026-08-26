### Added: `G1Driver.send_action` publishes `LowCmd_` on `rt/lowcmd` (write path lands)

The G1 native driver's arm write path is now live. `send_action` builds a
`LowCmd_` from the caller's `action` mapping and publishes it via the
`DDSPublisher` the driver constructs alongside its subscribers in
`connect_eagerly`:

```python
driver.send_action({
    "joints": [15, 16, 17, 18, 19, 20, 21],
    "q":  [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    "kp": [50.0] * 7,
    "kd": [1.0]  * 7,
})
# → {"status": "success", "content": [{"text": "wrote LowCmd_ on rt/lowcmd for 7 joint(s)"}]}
```

Every safety gate that already existed (`_check_motion_gates` with `"arm"`
scope; the battery-under-floor check) runs first, so an FSM outside
`HANDSHAKE_FSMS` or a battery below the floor still refuses without any
DDS touch. Shape errors in `action` (a length mismatch between `joints`
and `q`, an out-of-range motor index) refuse before the wire too, so a
partial write is impossible on a shape error.

The `LowCmd_` IDL is resolved lazily on the first `send_action` call and
cached, so a control loop at 500Hz pays the import cost once. No
`unitree_sdk2py` at driver module load, matching the invariant `DDSPublisher`
and `DDSSubscriberSet` already keep.

This ships the arm SDK primitive that issue #361's P0 unblock needs;
`start_task` / `run_policy` (the control loop that consumes this primitive
at 500Hz with per-step FSM re-gate and zero-torque on stop) is the next
PR in the same stack.
