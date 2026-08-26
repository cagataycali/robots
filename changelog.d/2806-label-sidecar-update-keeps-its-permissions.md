### Fixed: updating the label sidecar no longer narrows its permissions

`strands_robots.episode_labels._write_document` replaces the sidecar by writing
a `tempfile.mkstemp` file next to it and `os.replace`-ing it into place. A
`mkstemp` file is owner-only, so the rename carried `0o600` onto the
destination: a sidecar that arrived group- or world-readable came out of the
next annotation readable to nobody but the writer. That is the ordinary case
rather than an unusual one. The module docstring gives the sidecar's whole
purpose as travelling with the dataset directory, and every way it travels - a
copy, a `tar -x`, a Hub download, a clone - lands it at the reader's umask,
which is wider than `0o600` in at least one bit. Measured on a sidecar at
`0o644`, both `record_deterministic_verdicts` and `annotate_episode` left it at
`0o600` and reported success; the cost surfaced later and elsewhere, as a
`PermissionError` out of `read_labels` or `filter_episodes` on labels another
account could read a moment before.

The destination's mode is now read before anything is written and applied to
the temp file, so the sidecar is never momentarily readable to fewer callers
than it was - chmod-ing `path` after the rename would leave exactly that
window, and the ordering is graded rather than left to the next edit. A sidecar
this module creates keeps `mkstemp`'s owner-only mode: replacing content is not
the place that decides who may read a dataset, and this change does not widen
anything. That is the one way the construct departs from
`strands_robots.simulation.safe_output.atomic_write_bytes`, which holds its own
output to `0o600` on purpose because the sim output roots are private to the
running user - a reason that does not transfer to a dataset artifact, and the
departure is now written down at both ends instead of arising from `mkstemp`'s
default.
