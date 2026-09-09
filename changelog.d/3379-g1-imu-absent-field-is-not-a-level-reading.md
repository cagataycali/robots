### Fixed: a G1 IMU field the message does not carry is `None`, not a level reading

`G1Driver._on_lowstate` read its four `IMUState_` vectors with typed
defaults - `[0.0, 0.0, 0.0]` for `rpy`, `gyroscope` and `accelerometer`,
`[1.0, 0.0, 0.0, 0.0]` for `quaternion`. A `getattr` carrying a default
cannot fail, and every one of those constants is a well-formed reading of a
robot that is fine: zero rpy is perfectly level, the identity quaternion is
upright, and a zero accelerometer is free fall, which a standing robot never
reports because gravity always lands on one axis. A firmware that renamed or
dropped a field therefore published a constant shaped exactly like telemetry
- and kept publishing it, to `strands/{peer_id}/imu` via
`SensorLoopsMixin._read_imu`, for as long as the robot ran. A fleet reading
attitude off the wire was told a falling humanoid was level.

The four reads now go through `telemetry_float_list`, which is what the twin
driver already did: `Go2Driver._on_lowstate` reads the same four names that
way, so after the shared-coercer work this was the one Unitree telemetry read
still carrying typed defaults. The same rule is kept one method down in the
same class (`_on_bms` reads through `getattr(msg, name, None)` so a renamed
field lands `None` rather than a plausible zero), stated outright by the
sibling humanoid in `strands_robots.drivers.booster.parse_low_state` ("a
snapshot that reports a zeroed IMU the robot never sent is worse than one that
reports none"), and documented by the consuming `g1_imu` verb, which described
every field as "or `None`" and promised it "does not fabricate a reading the
driver does not have" while the writer made the per-field `None` unreachable
for any driver that had received a frame.

Because the fields are now coerced one at a time, a single unreadable vector
reports itself as `None` instead of raising inside its comprehension, being
swallowed by the callback's `except`, and abandoning the whole assignment -
which previously discarded the three good readings that arrived in the same
frame and left the last record in place as though nothing had been received.

No new coercion helper: the shared function was already imported in the
module, so this removes a hand-rolled read rather than adding a rule.
