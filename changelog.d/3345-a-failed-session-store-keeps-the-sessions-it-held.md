### Bug Fixes

- **tools**: A session store that cannot be written keeps the sessions it already held.
  `lerobot_teleoperate` and `lerobot_train` write the same whole-document store, so a
  write that landed partially left a prefix where the store was - and both load paths
  report an unparseable store as *no sessions*. One failed write therefore dropped every
  record at once, while the detached processes those records named kept running: `list`
  answered `status="success"` with no sessions and `stop` answered that the session was
  not found. The map is now serialized before the destination is touched and committed
  through a temp file plus `os.replace`, in one shared owner both tools ask through, so a
  full disk leaves the previous store intact and a record JSON cannot represent is
  refused naming the store rather than truncating it.
