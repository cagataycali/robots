### Fixed: a Reachy link bring-up that will not be adopted is released, not abandoned

`ReachyDriver._start_link` submits the link's `start` coroutine to a background
loop and waits for it. Both ways that wait can end badly leave the link
unadopted - `_link` stays `None`, so `cleanup` has nothing to stop and no later
verb can reach the link again - yet neither path stopped the link. `start` gets
far enough to open something in the ordinary case, not a narrow race:
`WebSocketLink.start` assigns its connected daemon socket before it spawns the
read task, and `ZenohLink.start` subscribes to its first topic before its
second. So every failed bring-up stranded a live reader for the life of the
process, writing sensor frames nobody reads - the outcome `connect_eagerly`
already refuses a second link in order to avoid.

The timeout half also reported no cause. `str(TimeoutError())` is the empty
string, so a handshake that outran the budget came back as `link to
reachy-a.local:8000 failed to start: ` - a reason that trails off after the
colon and names nothing an operator can act on. Measured against a link that
subscribes and then does not hand back, that reason arrived after 10s with the
link still subscribed, never stopped, and the interpreter reporting `Task was
destroyed but it is pending!`.

Both failure paths now cancel the handshake and ask the link to stop, on the
loop it was started on, before that loop is stopped; both concrete links
tolerate a `stop` after a partial `start`. The timeout is caught by name and
reports the budget that expired instead of an empty exception string, and the
budget is a module constant (`_LINK_START_TIMEOUT_S`) rather than an inlined
literal. A `stop` that itself fails is logged and still yields the named connect
failure, so teardown of a failed bring-up cannot raise through the driver's
no-raise contract. The healthy path is unchanged: the link is adopted, left
running, and its loop and thread published as before.
