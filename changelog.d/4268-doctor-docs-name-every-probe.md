### Fixed: the doctor docs now name every probe the CLI runs

`strands_robots.doctor.CHECKS` grew an `IoT Direct` row between `Mesh` and
`Sim Test`, but two docs pages that describe the doctor's checks
(`docs/start/doctor.md` and `docs/reference/cli.md`) stayed at 15 rows.
`--list` printed sixteen names, the report printed sixteen lines, but the
"The probes" table and the CLI reference's run-order prose named only
fifteen. A user landing on either page could not recognise the
`SKIP  iot direct: STRANDS_MESH_BACKEND=zenoh (no AWS IoT leg)` row that
appears on every non-IoT run, and had no next-step advice for its `FAIL`
verdict.

Both pages now name the probe (the sample report gains the `SKIP` line and
the table gains one row; the CLI reference gains the label in its run-order
sentence). `tests/test_doctor_docs_name_every_probe.py` pins the rule going
forward - it walks every `CHECKS` label and refuses one that neither docs
page names, so the next probe to land is graded on arrival.
