### Fixed: a mobile base does not move on a velocity command the bridge cannot honor

`RosBridgedRobot.drive` and `RtpsRobot.drive` are the calls that physically move
a wheeled robot, and neither validated any of the four knobs the command
carries. `duration` sizes the published burst as
`max(1, round(duration * publish_rate))`, so the floor turned a hold no message
count expresses into a single full-speed `Twist`:

```python
robot.drive(linear=0.5, duration=0)     # status 'success'
# -> one Twist published at 0.5 m/s: the base starts moving
robot.drive(linear=0.5, duration=-5)    # status 'success', same single command
robot.drive(linear=0.5, count=0)        # status 'success', nothing published
robot.drive(linear=float("inf"))        # status 'success', inf reaches cmd_vel
robot.drive(linear=0.5, duration=float("nan"))
# ValueError: cannot convert float NaN to integer  (raised past the result dict)
```

`use_ros` / `use_rtps` cannot cover this: they validate the topic and interface
type, but `duration` never reaches them (they receive only the derived message
count) and neither checks `count` or the Twist field values. Both `drive`
methods are also bound as agent tools, where raising escapes the
`{"status": ...}` dispatch contract.

`drive` now reports an error result - publishing nothing - for a `duration` that
is not positive and finite, a `count` that is not a positive whole number, and a
`linear`/`angular` velocity that is not a finite number (both signs stay valid;
reverse and clockwise are real commands). `publish_rate` is refused at
construction, where a bad topic already raises, because `drive` multiplies by it
and the transport paces at `1 / rate`, so a non-positive rate removes the pacing
instead of slowing it. The rules are the shared
`strands_robots.utils` validators - `positive_finite_number_error`,
`positive_whole_number_error`, and a new `finite_number_error` for a signed
physical quantity - so the two transports cannot drift apart. Every command that
worked before is unchanged, including a hold shorter than one publish period,
which still sends the command once.
