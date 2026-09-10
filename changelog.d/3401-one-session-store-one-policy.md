### Fixed

- The detached-session store now has one reader, so it has one retention policy.
  `lerobot_train` and `lerobot_teleoperate` deliberately keep every robot session
  in one JSON document, but each defined its own `SessionManager` over it and the
  two disagreed about what a read may delete: the training store kept every
  record, while the teleoperation store dropped any whose pid did not answer as
  running and wrote the pruned map back to disk. Whichever reader deletes a
  record deletes it for both, so one `lerobot_teleoperate(action="list")` erased
  records the training tool had documented itself as keeping - and that store is
  the only place a detached run's pid is written down, so those runs could no
  longer be stopped through the tool. `SessionManager` and `SESSION_DIR` now have
  a single definition in `strands_robots.tools._process_stop`, retaining every
  record until `remove_session` is asked; `list` and `status` still derive
  running from the pid at the moment they are asked.
