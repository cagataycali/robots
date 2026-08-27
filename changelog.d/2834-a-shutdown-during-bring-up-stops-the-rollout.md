### Fixed: a `cleanup()` that lands during bring-up abandons the rollout instead of finishing it

`cleanup()` sets `_shutdown_event` and only then calls `stop_task()`, gated on
`status == RUNNING`. A teardown arriving while the task is still in `CONNECTING`
- a motors-bus handshake plus per-camera warmup, seconds on a real arm and
longer on a multi-camera rig - therefore sets no stop latch at all, and the
shutdown event is its only record. `_execute_task_async` checks
`_honor_stop_request()` at each stage boundary and that gate read only the stop
latch, so both checks passed and the rollout went on to finish the bring-up it
had just been asked to abandon.

The reported status was already right. `_shutdown_error` refuses a task started
*after* `cleanup()` rather than merely relabelling it, and its docstring says
why: "Bring-up is not free of side effects, which is what makes this a refusal
rather than a cosmetic status fix." It then lists them. The in-flight producer
reached the same ones, and the terminal-status discriminator - which does read
both latches - could only correct what the rollout said about itself, not what
it did. Measured on the two-device double, with `cleanup()` landing while the
bring-up is parked inside `connect()`:

```
                                              before      after
policy-server dial / checkpoint load          1           0
observation read off the bus and cameras      1           0
Policy.reset() on the caller's own object     1           0
cleanup() blocked on that work (start_task)   0.703s      0.402s
task status / steps                           stopped / 0 stopped / 0
arm commanded                                 no          no
```

The reset is the sharpest of them: a caller may drive one policy object through
several tasks - the documented `run_policy(policy_object=...)` reuse pattern -
so clearing its action-chunk cache and sampler RNG while backing out of a
cancelled rollout corrupts a rollout running elsewhere. The sibling producer's
`reset_calls == 0` was already pinned; this one's was not.

Three readers decide whether the rollout in flight may continue, and they must
agree: the bring-up gate, the control loop's exit condition, and the terminal
discriminator. The last two read both latches and the first read one. Asserting
their values agree cannot see that - two copies of the same disjunction agree up
to the moment one is edited, which is how the drift arose - so all three now
read a single `_rollout_stop_latched` predicate, and a test walks the module to
check the disjunction has exactly one owner.

Widening `cleanup()`'s own `stop_task()` guard to cover `CONNECTING` would also
stop the side effects, and is ruled out rather than merely disfavoured: it trips
`test_a_shutdown_during_bring_up_reports_stopped_not_completed`, which asserts
`_stop_requested.is_set() is False` after that teardown. Reading the latch at
the gate also covers any setter of `_shutdown_event`, not just that one branch.

Unchanged: an operator `stop_task()` during bring-up behaves exactly as before,
a healthy rollout still resets its policy once and reads its observations, and
the terminal discriminator remains the backstop for a shutdown landing after the
last stage check. Deliberately out of scope and pinned as such: the devices the
parked `connect()` opens are still open afterwards. That call is already inside
its blocking handshake when the teardown lands, so it completes either way, and
for a rollout `cleanup()` did not submit it completes after `_disconnect_devices()`
has run. Closing them needs `cleanup()` to wait for a rollout running on a
caller's thread, which is a different fix.
