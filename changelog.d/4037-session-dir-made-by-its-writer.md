### Fixed: importing a session tool no longer creates a directory in the caller's working directory

`strands_robots.tools._process_stop` created `$PWD/.strands_robots/.sessions`
while its module body ran, so `import strands_robots.tools.lerobot_train` (or
`lerobot_teleoperate`) wrote two directories beside any caller that merely
loaded the tool, and raised `PermissionError` from the import statement where
that working directory is read-only - a failure no handler around a session verb
can answer. The two doors that write into the directory create it now:
`session_log_path`, whose every caller opens the path it returns, and
`store_sessions`, which no longer requires its destination's parent to exist.
