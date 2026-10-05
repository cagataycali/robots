### Added: the Unitree H1-2 is driven natively, on the G1's `unitree_hg` frame behind the Go2's release gate

`Robot("h1_2", mode="real", port="<robot IP>", network_interface="eth0")` now
builds `Go2Driver` instead of refusing the robot: lerobot registers no H1-2
type, so it was simulation-only. The vendor's H1-2 low-level example releases
the onboard motion mode and then streams `unitree_hg` `LowCmd_` frames, so the
H1-2 is a `WireProfile` with `idl="unitree_hg"`: its 27 body joints are keyed by
the MuJoCo model's names onto the SDK's `H1_2_JointIndex` slots with the
example's gains (`kp=100` legs and torso, `kp=50` arms, `kd=1`), each frame
carries `mode_pr=0` and echoes the `mode_machine` read from `rt/lowstate`, and
the battery floor reads `rt/lf/bmsstate`. A write refuses until `mode_machine`
has been read; the 24 hand joints of the model are refused, not written.
