### Fixed: a teardown that could not halt the robot says so

`BoosterDriver.cleanup` called `stop_task()` and `CrazyflieDriver.cleanup` called
`land()`, and both dropped the envelope those verbs return. `cleanup` is annotated
`-> None`, so nothing anywhere - envelope, flag or log - recorded that the halt had
not landed, and the hook then went on to close the SDK channels and the radio link
a retry would have needed. A T1 whose halting twist the locomotion controller
refuses keeps walking at its last commanded velocity, and an aircraft whose
descent is refused stays airborne, with nothing left in the process able to reach
either.

Both now read the verdict through `halt_failure_detail` and log it at `error`,
naming the robot, which half did not land and that the channels are going - the
shape their own `stop` hooks already had. The release stays unconditional: a
teardown that stopped half-way would leak the channel *and* leave the robot
moving. `HardwareDriver.cleanup` documents the obligation beside `stop`'s.

`tests/drivers/test_a_verdict_free_hook_logs_the_halt_it_could_not_complete.py`
graded the relation on `stop` alone. It now grades both hooks the driver protocol
annotates `-> None`, so the next driver is graded on arrival. The wire-failure
relation deliberately stays on `stop`: what the `cleanup` hooks catch is a
channel close during teardown, not a halt.
