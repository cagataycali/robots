### Tests: the posture-flag sweeps walk their value tables in one cell

The `run_policy` posture-flag suite and the `lerobot_teleoperate` flag-domain
suite gave every probe value (each truthy spelling of off, each falsy
non-boolean, each numpy boolean) its own cell on every surface. Each test now
walks the table in one cell and names the failing value in the assertion. The
two files go from 363 cells to 118 with the same package lines executed.
