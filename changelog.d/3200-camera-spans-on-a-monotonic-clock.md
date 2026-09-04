### Fixed: a camera tool's reported span is the work's, not the size of a clock correction

Four of `lerobot_camera`'s five span-reporting handlers measured those spans on
`time.time()` -- `capture`'s connect and read times, `capture_batch`'s per-camera
time and its batch total, `record`'s achieved duration, and `test`'s connect plus
its two ten-frame read windows. `preview` already measured its bases on
`time.monotonic()`, so the module held both answers at once.

`time.time()` is the current opinion about the date, so an NTP correction, a
`date -s` or a resume from suspend landing inside one of those windows is
subtracted from the span, and nothing raises. On a camera reading in 20 ms, with
one correction placed inside the ten sync reads of `action="test"`:

```
+30 s   Average: 3.020s    Est. FPS: 0.3     Sync capture: Slow
-30 s   Average: -2.980s   Est. FPS: -0.3    Sync capture: Good
```

The second line is the worse one. `test` does not merely report the span: it
turns it into a claim about the camera -- `Est. FPS` is `1 / avg_sync_time`,
`Sync capture` is `Good` below 100 ms, `Connection` is `Fast` below one second --
and a negative average is below every one of those thresholds. So a corrected
clock reported a physically impossible frame rate and called the device good,
while an average of exactly zero divided by zero. `capture` reported
`Connect time: -29.980s` on the same step, and `capture_batch` reported
`Total time: 30.041s` for a batch of one 20 ms read.

Every duration base now reads `time.monotonic()` and carries that clock in its
name, and `capture_batch` no longer spells its base and its duration as one
variable. The absolute stamps this tool writes -- a filename's date, a report's
`Timestamp` line -- come from `datetime.now()` and are unchanged; the module
states that boundary once, in its docstring, and now reads no wall clock at all.

The package-wide scan in `tests/test_expiry_gates_survive_a_clock_step.py` was
silent on all of this and was right to be: it grades a wall-clock read that
decides whether to keep *waiting*, and none of these does -- the recording is
bounded by a frame count and the windows by `range(10)`. The new cells in
`tests/tools/test_camera_durations_survive_a_clock_step.py` grade what the tool
*reports* across a step instead.
