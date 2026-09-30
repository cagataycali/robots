### Fixed: the doctor page lists all sixteen probes, including `IoT Direct`

`strands-robots doctor` prints an `IoT Direct` row between `Mesh` and
`Sim Test`, but `docs/start/doctor.md` described only the other fifteen, so the
`SKIP  iot direct: ...` line every non-IoT run prints had no description and
its `FAIL` no advice. The probe table and the sample report now carry it, and
the page no longer says no probe touches the network: a configured
`IoT Direct` makes one HTTPS call. `docs/reference/cli.md` drops its own copy
of the probe list, which had drifted the same way, and links to the doctor
page instead.
