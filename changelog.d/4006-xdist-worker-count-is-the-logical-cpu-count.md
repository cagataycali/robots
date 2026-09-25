### Fixed: the distributed suite asks for one worker per logical CPU, so a one-core/two-thread runner gets two

The `test` script's `-n auto` resolved through `psutil.cpu_count(logical=False)`
- the physical core count, because psutil is importable in the test env through
the `lerobot` extra - and the `ubuntu-latest` pool serves the same 2-vCPU size as
two-core VMs and as one-core/two-thread VMs. On the second kind `auto` was one
worker: measured on one such runner over 9,456 tests with coverage on, `-n auto`
created `1/1 worker` and took 249.08 s at 48% CPU, `-n logical` created `2/2
workers` and took 118.57 s at 101% CPU. At the full suite's 1,741-1,784 s on two
workers that is the 60-minute reap band again, on whichever runs land on that
kind. The script now says `-n logical`, which reads the CPUs the kernel
schedules on and is 2 on both kinds; `loadfile`, the `addopts` split and the
single-process `test-integ` are unchanged. The pin asserts `logical` and carries
the measurement in its message.
