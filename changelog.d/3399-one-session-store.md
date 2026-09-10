### Fixed: one session store for `lerobot_teleoperate` and `lerobot_train`

Both tools record their detached children in the same `active_sessions.json` on
purpose, but each carried its own `SessionManager` over it and the copies
disagreed. One pruned finished records **on every read** and wrote the pruned map
back, so a read-only `lerobot_teleoperate(action="list")` erased the other tool's
records - including one whose pid could not be inspected, which is the only
handle on a run that may still be live. Both were load-modify-write with no lock,
and the shared store's temp file was named from the store alone, so two writers
could interleave into it and commit a document neither wrote - which the load path
reads as *no sessions*, losing every recorded pid at once.

There is now one `SessionManager` (`strands_robots.tools._session`) with one
policy: a read never writes, the single prune runs inside `add_session` and drops
only a record whose process is provably gone, and the load-modify-write is guarded
by `fcntl.flock` on a lock file beside the store. `store_sessions` names its temp
file for the writing process. The session directory is created at the first write
instead of at import, so importing either tool no longer creates
`./.strands_robots/.sessions` in whatever directory the process happens to be in.
