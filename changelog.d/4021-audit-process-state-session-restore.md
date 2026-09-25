### Fixed: the safety audit log's process state is restored between tests

`strands_robots.audit` keeps four pieces of process state - the per-peer sequence
counters and the three one-shot flags on `_AUDIT_STATE` (`seq_loaded`,
`audit_log_seeded`, `psk_fingerprint`). All four outlive the test that fills
them, so 25 test modules reset some of it in a fixture of their own, and no two
reset the same set: 11 left `psk_fingerprint` out, 2 left the counters out, 2
left `audit_log_seeded` out. Measured over those modules in one process, 11 of
the 24 module boundaries handed the next module dirty state and 91 of 259 tests
finished non-pristine - and a leaked fingerprint is not a stale value but a
refusal, since the next write under a different `STRANDS_MESH_AUDIT_PSK` is
rejected as a mid-run rotation and replaced by a `PSK_DEGRADED` poison record.
`tests/conftest.py` restores it now, beside the four process-globals the session
already owned, and the callers lose their copies; resets inside a test - the
simulated fresh process, a flag pinned `True` to keep the log walk out of the
subject - stay, because those are the test's own subject.
