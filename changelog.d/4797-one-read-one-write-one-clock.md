### Changed: a run reads the body once per tick, never bursts after a stall, and keeps one clock

`Robot.run` and `Simulation.run` now share one step: one `observe()`, one decision, one write. A tick used to
read the body three times (25 ticks cost 77 reads); joints a policy leaves out now hold the observation the
tick already took (`coerce(measured=)`), and `Robot.command(targets)` writes without reading. Paced runs
(hardware, `realtime=True`) space writes at least one period apart; a late write is counted in
`RunResult.deadline_misses` / `worst_late_s` and re-phased instead of followed by a burst of stale commands,
and `run(on_miss="fault")` ends the run there with `stopped_by="deadline"`. A hardware robot takes
`Robot(clock=)`; `robot.clock` (the world's time in a world) stamps `Observation.t`, the e-stop latch's
`engaged_at`, the watchdog and `RunResult.seconds`; `RunResult.wall_seconds` is the host's time.
In lock-step one robot's fault or e-stop ends the tick for every robot, and `lockstep(stop_event=)` stops it.
