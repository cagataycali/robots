### Fixed: the Go2 write gate shuts on a mode reading that says the robot is not free

`Go2Driver._sport_mode_released` IS the write gate - `_check_motion_gates`
reads that cached boolean rather than taking a DDS round trip, so nothing
else re-asks the robot. It was cleared when a `CheckMode()` reading could
not be decoded, but not when a reading decoded to the name of a motion mode
still holding the legs. A Go2 that re-entered a mode after being released
(the app, a fall-recovery, an operator's remote) therefore kept a gate that
an earlier release had opened: `release_sport_mode()` correctly refused with
"sport mode 'ai' still active after 2 release attempts", and the very next
`send_action` was admitted and published `rt/lowcmd` into a fight with the
onboard controller over twelve motors. The stale verdict was reported as
`state["sport_mode_released"] is True` as well.

Every reading other than `""` now clears the flag, so the gate follows the
last reading rather than the first success. An empty mode name - the one
reading that is evidence the robot is free - still opens it.
