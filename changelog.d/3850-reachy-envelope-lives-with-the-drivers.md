### Changed: the Reachy motion envelope lives with the two drivers that share it

`MOTION_ENVELOPE_DEG`, `HEAD_BODY_YAW_DELTA_LIMIT_DEG` and `envelope_error` moved
from `strands_robots.tools.reachy` to `strands_robots.drivers.reachy_envelope`.
Their only two consumers are drivers -- `ReachyDriver.send_action` and the Device
Connect driver's movement RPCs -- and no `reachy_*` agent verb reads them, so
both drivers had to import the agent-tool package to bound a joint. The limits,
the refusal wording and every branch of `envelope_error` are unchanged; the
package's inversion roster loses two entries (11 upward runtime edges to 9).
Anything importing these three names from `strands_robots.tools.reachy` reads
them from `strands_robots.drivers.reachy_envelope` instead.
