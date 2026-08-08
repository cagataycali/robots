### Fixed: `uv.lock` no longer resolved `pyproject.toml`, and nothing checked

The lock was last written on 2026-07-25. `pyproject.toml` was edited on three days
since without a relock, so `uv lock --check` failed on `main` and a locked install
resolved a dependency set the manifest forbids. Nothing compared the two files, and
CI stayed green throughout.

Four rows had drifted. `lerobot` was pinned at `0.6.0` against a declared
`>=0.6.1` floor - the floor that exists so `stream_dataset(repo_type="bucket")` is
resolver-guaranteed rather than only documented, so the lock guaranteed its
opposite. `roslibpy` was absent from the lock entirely and the recorded
`[rosbridge]` extra was empty, so that extra and `[all]` resolved without the
package `use_rosbridge` imports. `mink` and `qpsolvers` were missing from
`[sim-mujoco]` and `[sim-newton]`, leaving a locked sim install without the IK
stack `move_to` needs. And the `huggingface-hub` floor was recorded one raise
behind.

The lock is regenerated, and the parity it had silently lost is now asserted rather
than assumed. `scripts/check_lockfile_parity.py` compares the manifest against the
transcription uv records in the lock's own `[package.metadata] requires-dist`, and
compares each declared specifier against the version the lock pins; both run
offline, so the guard is an ordinary test in the required suite rather than a
resolver invocation. It mirrors two encodings rather than approximating them:
self-references are expanded transitively, because uv records the closure, and a
name may be pinned more than once, because `[tool.uv] conflicts` forks the
resolution.

The dependency-audit suite is where this would have been caught, and shows why it
was not: it names `uv.lock` twice in prose as the artifact whose regressions it
protects, while every assertion in it reads only `pyproject.toml`.
