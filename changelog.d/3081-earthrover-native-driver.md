### Added: native driver for the EarthRover Mini Plus

`Robot("earthrover", mode="real")` could not be built at all: the rover has a
`lerobot_type` for dataset purposes but no lerobot robot class, so the default
driver raised "Unsupported robot type". `EarthRoverDriver` drives it through
the vendor's `earth-rovers-sdk` - a local HTTP service (default
`http://localhost:8001`) that proxies to the rover over WebRTC - and the
registry entry declares `hardware.driver="strands"` so no `driver=` keyword is
needed.

A rover is velocity-commanded: unlike an arm it does not hold still when you
stop talking to it, and whether the firmware times a twist out on its own is
not documented by the vendor. So teardown is a stop first - `cleanup()` sends
a best-effort zero twist before releasing the session, and `stop_task()`
returns the zero-twist send's own envelope, so a caller learns whether the
stop *reached* the SDK rather than that a flag was cleared. The parting
twist's errors are swallowed by design: a dead link must not block the close,
and a rover behind a dead link cannot hear a stop anyway.

The write path is one twist per `send_action` call - `linear` and `angular`
clamped to `[-1, 1]`, an optional `lamp` flag, an absent axis commanded `0.0`
because a twist is a complete statement of intent ("turn" also means "stop
driving forward"). An unknown channel is refused naming the valid set, a
non-finite axis is refused through the shared `finite_number_error` domain,
and `port=` refuses a filesystem path by shape with a sentence saying that
spelling belongs to the serial arms - `port` is polymorphic across drivers,
and the wrong shape should fail at the chokepoint, not one call later as
"No host supplied". A rover observed turning the wrong way is corrected with
`turn_sign=-1` at construction, visible at the call site, not an environment
variable.

Reads degrade rather than raise: `read_state()` polls `/data` fresh and falls
back to the cached snapshot when the poll fails, `get_observation()` is `{}`
always (a wheeled base has no joints, and publishing wheel RPMs as "joints"
would put velocities where every consumer expects positions), and
`is_connected` derives from the session leaf so a torn-down driver cannot
report live. Camera frames come from `/v2/front` and `/v2/rear` with the
format read off the payload's own magic bytes rather than assumed.

`requests` joins the mypy override list: it was already a declared dependency,
but this is the package's first direct import of it, and it ships no stubs.
