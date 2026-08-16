### Fixed: `use_ros` / `use_rtps` refuse a numeric option the transport cannot honor

The `count`, `rate` and `timeout` options of both ROS 2 transport tools were
forwarded unvalidated. `rate=0`, a negative rate, `nan` and `inf` all fell
through `period = 1.0 / rate if rate > 0 else 0.0` to an unpaced `period = 0.0`,
so `publish` sent the whole burst back-to-back and still reported success - a
velocity hold collapsed into an instantaneous burst that a base latches as its
last command. `count=0` and `count=-1` published nothing and reported
`published -1 message(s)`; `count=True` published one; `count=2.7`, `"3"` and a
non-numeric `rate` surfaced a raw `range()` or comparison `TypeError` naming
neither the tool nor the option. On `echo` a non-positive or `nan` `timeout`
returned zero samples as a success and `inf` never expired.

Each option an action actually consumes is now checked against the shared domain
for its kind (`positive_count_error` for a `range()` bound, `positive_finite_number_error`
for a rate or a span of seconds), alongside the existing topic/type allowlist and
ahead of the backend probe - so the refusal happens before any DDS entity joins
the graph, reports identically with or without a ROS 2 distro installed, and is
the same on both transports. An option the requested action never reads is not
second-guessed.
