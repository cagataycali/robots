### Fixed: every door that accepts a teleoperator refuses one it cannot poll

A teleoperator is consumed identically wherever one is accepted: a background
loop calls `get_action()` once per tick and forwards the result. Only the local
attach door graded that contract. `Robot.start_teleop_publish` and
`InputPublisher` did not, so a device with no callable `get_action` was started
rather than refused - the entry point returned `status="success"` with the topic
and the `start_teleop_receive(...)` call for peers to subscribe with, while the
loop it started raised `AttributeError` every tick (measured: 70 errors, 0 frames
on the wire in 0.35 s at 200 Hz), reported `running`, and fell silent once its
logging budget was spent.

Worse, that entry point stops a publisher already registered under the same
`device_name` before constructing the new one, and its own comment gives the
reason the other two arguments are graded ahead of that teardown: a rejected call
must not stop a live stream. A device that could never be polled therefore cost
the caller the working publisher it was replacing. `InputPublisher` accepted it
at construction too, though the same constructor already refuses an unusable
`hz` for exactly this reason - the loop it feeds runs on a thread where the
mistake surfaces as a dead publisher that still reports `running`.

All three now read one shared domain, `utils.teleoperator_contract_error`, so a
device one door turns away cannot be started by the next. `attach_teleop`'s
verdict is unchanged; the two mesh doors moved onto the answer it already gave,
and its inline check became a call to the shared helper so the wording has one
owner. A non-callable `get_action` attribute is refused as well as a missing one.
