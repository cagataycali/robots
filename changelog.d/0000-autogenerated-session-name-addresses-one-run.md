### Fixed: an auto-generated session name addresses the run it was handed back for

`lerobot_train` and `lerobot_teleoperate` derived an unnamed session's name from
a second-resolution clock (`f"train_{int(time.time())}"`), so two `start` calls
in the same second derived the same one. That is the ordinary case for an agent
issuing independent tool calls together, not an exotic race: two concurrent
training launches on one machine produced

    start #1 -> {"session_name": "train_1789213295", "pid": 13457}   # diffusion
    start #2 -> {"session_name": "train_1789213295", "pid": 13456}   # act
    list     -> 1 session: train_1789213295 -> pid 13456 (act)

Both processes ran. The second record replaced the first, so the name `start`
returned for the diffusion run afterwards addressed the act run: `status`
reported a stranger's policy and output directory, `stop` would have signalled a
stranger's process and reported success, and the diffusion run was reachable by
no name at all. Both also appended to the single log file the shared name
resolves to, interleaving two runs' output in the tail `status` reports.
Serialized instead of concurrent, the same collision refused the second launch
outright - `Session 'train_1789213295' already exists`, naming a session the
caller never supplied.

`strands_robots.tools._process_stop` already refuses that outcome when it
arrives through a reused pid: a record pointing at "whatever holds the number
next" reports a stranger as the session "and the `stop` verb that verdict
invites signals it". A reused name reaches the same place.

Both tools now name an unnamed session through one `generate_session_name`
helper, which keeps the timestamp for readable, sortable names and appends
random entropy so the name a caller was handed keeps pointing at the run it was
handed for. Entropy rather than checking the store first, because two concurrent
`start` calls can both pass that check before either writes. A session name the
caller supplied is untouched: they own the name, so a collision with one they
already hold is still reported rather than silently renamed.
