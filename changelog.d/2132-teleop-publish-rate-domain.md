### Quality: pin the rate domain on the third teleop entry point

`Robot.start_teleop_publish` refuses an `hz` its publish loop cannot honor -
`1 / hz` is the period - and the rate-guard suite's own docstring names it as one
of the three entry points sharing
`strands_robots.utils.positive_finite_number_error`. It was the one the suite
never drove: the `teleoperate(publish=True)` tests reach the mesh publisher
through a stand-in host whose `start_teleop_publish` records the call and returns
success without validating anything, so the real method's refusal had never run.

That refusal also sits ahead of the teardown of any publisher already registered
under the same device name, and its comment states why - a rejected call must not
stop a live stream. The identifier half of that ordering contract was pinned; the
rate half was not.

Five tests close it: every unusable rate is refused in the shared domain's exact
words, a refused rate registers no publisher and publishes nothing, a live
publisher survives a refused rate, an accepted rate does replace and stop it (the
mirror, without which nothing distinguishes "the guard runs first" from "there is
no teardown"), and the entry point and `InputPublisher` agree on every value.
