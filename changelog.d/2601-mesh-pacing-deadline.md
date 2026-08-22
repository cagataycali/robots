### Fixed: a publish loop's period is a deadline now, not a delay

Thirteen loops in the mesh paced themselves with `self._stop_event.wait(period)`.
The shape was right -- it sleeps for the period and returns early on stop, so
shutdown never waits out a tick -- but a `wait(period)` is a delay where a rate
needs a deadline. The time a tick spent reading a bus or grabbing a frame was
added to the period instead of subtracted from it, so a loop achieved
`1 / (period + work)` and every counter reported that as the rate the hardware
managed.

Measured on Linux against an idle epoll timer, so this is the floor rather than a
platform quirk: 50Hz with a 10ms tick body achieved 31.92Hz, 30Hz with a 10ms
body 22.97Hz, 10Hz with a 40ms body 7.12Hz. Driving the real `Mesh._camera_loop`
for one second at a requested 30Hz with a 10ms frame grab captured 23 frames
instead of 30, at a 43.5ms median gap where the period is 33.3 -- and the rate a
recorded dataset's video was captured at is the rate the loop achieved, not the
`hz` the run reported. A second cause stacks on top on a host that inflates
`nanosleep`-family waits under background-QoS timer coalescing, where a nominal
100ms `Event.wait` costs ~247ms and even an empty 10Hz body reaches 4.33Hz.

`strands_robots.mesh.pacing.Ticker` waits on the selector timer instead. The
period is a deadline; missed deadlines are dropped rather than chased, because a
burst of frames stamped microseconds apart is a worse lie about a camera than a
gap; and a stop is honoured within a 10ms slice rather than at the end of a tick,
with `wake()` able to interrupt one immediately. `wait()` keeps `Event.wait`'s
sense -- `True` means stop -- so converting a loop cannot invert a shutdown test
by accident.

Converted: the three `Mesh` loops (state, heartbeat, camera), all seven
`SensorLoopsMixin` loops through one shared `_paced` generator, `InputPublisher`,
the teleop apply loop and the RTPS command poll. The input publisher and the
teleop loop already subtracted their own body time, so for those two the change
is single-ownership of that arithmetic plus the coalescing penalty rather than a
rate gain on an honest clock. Two waits are deliberately left alone and carry the
reason each rests on: a one-shot shutdown wait, and an exception-path backoff
where waiting longer than the period is the intent.

`sleep_penalty_s()` is exposed so a test can calibrate against the host it runs
on instead of loosening a number until it passes, and an inventory test scans the
package for the shape -- any `.wait(...)` on an attribute whose name reads as a
stop flag, in any statement position -- so "no pacer was missed" is checked
rather than claimed. That breadth is load-bearing: the teleop loop's event is
named `_teleop_stop_event`, and a `_stop_event`-only scan misses it entirely
while its eleven siblings look fixed.
