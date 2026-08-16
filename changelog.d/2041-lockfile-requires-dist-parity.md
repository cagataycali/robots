### Added: the required check now reports every way `uv.lock` can stop transcribing `pyproject.toml`

#2039 added the `uv lock --check` gate, which is authoritative but **advisory** --
the `default` ruleset lists exactly one required check. Its offline half ran
inside the required check but asserted two *properties* of the lock rather than
comparing it: every declared distribution present somewhere, and no locked version
below its declared floor. Against `main`'s stale lock that reported 2 of the 5
rows that had actually drifted.

The three misses were drift in the recorded set's shape rather than in a property
of a pin. `mink` and `qpsolvers` were both locked -- reachable via
`[cosmos3-sim]` -- so a presence check passed while `[sim-mujoco]` was recorded
as `['imageio', 'imageio-ffmpeg', 'mujoco', 'robot-descriptions']`: a locked sim
install with no IK stack behind `move_to`. `[rosbridge]` was recorded empty. And
`huggingface-hub` is pinned at `1.20.1`, which satisfies its `>=1.5` floor, so
only the recorded specifier was stale at `>=1.0` -- invisible to any assertion
about a correct pin.

`scripts/check_lockfile_parity.py` compares the whole declared set against uv's
own transcription of it in the lock's `[package.metadata] requires-dist`, in both
directions, offline -- no resolver, no network, no `uv` binary -- so it runs in
the required check. It reproduces the current lock exactly (111 declared rows
against 111 recorded, zero differences) and reports 24 findings against the stale
one, covering all five rows plus four the audit had not enumerated. Findings are
grouped by `(extra, name, extras)`, so a changed version range is one
`specifier-drift` finding naming both sides rather than two opaque rows with
different remedies.

The script exits 0/1 and can be run before pushing; `tests/test_lockfile_parity_requires_dist.py`
pins the live pair, the five repaired rows individually, the two encodings the
comparison depends on, and planted manifest/lock pairs for every finding class so
an empty finding list means agreement rather than blindness.
