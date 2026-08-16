### Fixed: pose_tool emergency_stop de-energizes the arm instead of reporting that it did

`pose_tool(action="emergency_stop")` returned
`{"status": "success", "text": "Emergency stop executed (torque disabled)"}`
while executing no code at all:

```python
if action == "emergency_stop":
    # This would require torque disable in real implementation
    return {"status": "success", "content": [{"text": "Emergency stop executed (torque disabled)"}]}
```

The bus was never opened and no packet was ever written, for any port,
connected or not. An operator or agent that reached for the one action meant to
stop a moving arm was told the arm had been released when nothing had happened
to it.

The handler now connects and writes `Torque_Enable = 0` (address 40 on the
Feetech STS/SMS control table) to every configured motor via the new
`MotorController.disable_torque()`, which attempts every motor even after one
fails and returns the ones still driven. An unreachable bus or any failed write
is reported as `status="error"` naming the joints that are still energized and
pointing at the hardware cutoff. The success text now also states that the arm
goes limp and drops what it is holding, which is what de-energizing means and
is not what "stopped" implies.
