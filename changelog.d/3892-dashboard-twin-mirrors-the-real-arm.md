### Added: the dashboard twin follows the real arm, reading the bus and never writing it

An operator with an SO-101 on the bench moved it by hand and expected the twin
on the Sim tab to follow. It now can: pick the arm's serial port instead of
**simulate** and the session that starts is a **mirror** - a thread reads
`Present_Position` from every motor at ~20 Hz and the model is posed from the
readings, so the render and the browser twin show the physical arm where it
is. No physics steps, no `Reset`, `joints` answers `400`: the arm decides.

The reader writes nothing. lerobot's bus is opened for the handshake and closed
with `disable_torque=False`, because the default close writes `Torque_Enable=0`
to every motor. Angles are the uncalibrated estimate `(ticks - 2048) · 2π /
4096`, labelled as such in the snapshot's new `bus` field next to the raw
ticks, read rate and age. A bus that stops answering shows `stale`, then
`error` with the reason. A pose outside the model's joint ranges is refused
whole, so the twin would otherwise freeze while the bus still read at 20 Hz:
that tick shows `refused` naming the joint, a state that ends nothing and
clears itself, so the telemetry socket survives it. A port that will not open
is a `502` naming it and
nothing is left holding the device - nor when the port opens but the engine
behind it fails to build or render. `GET /api/sim/ports` lists what a mirror
could read, servo buses first, opening nothing.
