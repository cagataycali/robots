### Fixed: the EarthRover `sensors` summary reads the headlamp as a flag

`telemetry_summary` rendered the lamp with `'on' if data.get('lamp') else 'off'`,
the one telemetry field in the drivers package read for truthiness. The same
module's write path holds `lamp` to `boolean_flag_error` precisely because a
word must not switch a headlamp, so the two doors disagreed about the same
field: `lamp="off"` was refused on the way out and read as *on* on the way
back. A snapshot whose firmware no longer carries `lamp` reported the headlamp
*off* - a state this function decided rather than one the rover described,
while `battery`, `signal_level`, `orientation` and `speed` all read `?` absent
and `get_status` published the field as `None`.

The SDK carries the lamp as the `1`/`0` `send_action` writes, so those two
integers and the two booleans are the readings; anything else now reads `?`
alongside its absent siblings. The four real readings render exactly as before,
and the whole `/data` snapshot the `sensors` verb publishes beside the summary
is untouched, so a caller that wants the raw field still has it.
