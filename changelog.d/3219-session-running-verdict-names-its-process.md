### Fixed

- `lerobot_train` and `lerobot_teleoperate` now answer "is this session running"
  about the process the record was written for, not about whatever holds its pid.
  Both stores persist a detached session's pid and outlive the run that made it,
  and both derived the running verdict from `psutil.pid_exists` alone - so once
  the kernel handed the number back out, a finished session read as running and
  its `stop` verb sent SIGTERM and then SIGKILL to an unrelated process.

  The guard both stores documented for this, `Process(pid).is_running()`, cannot
  be one: psutil records the creation time when the object is constructed, so an
  object constructed to ask the question agrees with whatever the pid means now.
  Each record now carries the identity of its process - how long after boot that
  process started, a duration rather than a date, so it survives the hours and any
  wall-clock correction between the run that writes it and the run that reads it -
  and `stop` refuses to signal a pid that no longer holds it, reporting
  `pid_reused` instead. A record written before this change carries no identity
  and is still answered by existence, so no session becomes unstoppable.
