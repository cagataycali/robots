### Added: `IsaacConfig` forwards Kit settings, caps Kit's task threads, and can bound a hung start-up

Several Isaac processes starting at once on one host could hang inside
`SimulationApp` start-up with no end: each Kit starts one `carb.tasking` worker
per CPU core, and on a 32-core L40S host 3 of 6 processes started together were
still booting after 240 s. The documented escape hatch, `IsaacConfig.extra`, was
never forwarded, so the Kit setting that avoids it could only be passed through
a private function.

`IsaacConfig` gains `kit_args` (Kit command-line settings, forwarded to
`SimulationApp` as `extra_args`), `task_threads` (the `carb.tasking` thread
count; with `task_threads=4` 12 of 12 such processes booted in 15-16 s), and
`boot_timeout_s`: a start-up that has not finished by then cannot be interrupted
from Python, so the process writes every thread's stack to stderr and exits
with status 70 instead of blocking forever. All three default to Kit's own
behaviour.
