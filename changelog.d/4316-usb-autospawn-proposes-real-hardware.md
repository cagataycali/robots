### Fixed: USB auto-spawn proposes a remembered real arm to the operator instead of starting it on a serial alone

The watcher matched a plugged-in board to a saved profile on the serial number
the device reports about itself and started a `mode=real` child under the
remembered peer id and calibration with nobody asked; a board with no serial was
keyed on its `/dev` path (f014, CWE-290). A real-mode match is now a pending
proposal, in the poll result, the activity trail (`autospawn_proposed`) and
`GET /api/devices/profiles` (`autospawn_pending`), naming the serial, vid:pid
and path seen and the robot it would become; the operator confirms it through
`spawn-remembered`, or lists the serial in
`STRANDS_DASHBOARD_AUTOSPAWN_REAL_SERIALS`, in which case the board's vid:pid
must match what `remember_profile` now records. Sim profiles still come up on
their own. A board without a serial, or two boards with one serial, starts
nothing and is written to the trail; a peer held back because its id is already
running or on the mesh is announced once at info level instead of debug.
