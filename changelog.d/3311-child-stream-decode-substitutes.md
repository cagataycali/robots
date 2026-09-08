### Bug Fixes

- **subprocess**: 25 sites read a child process's stream in text mode, which
  decodes with `errors="strict"`, so one byte the codec could not decode
  replaced the whole captured stream with `UnicodeDecodeError`. A child's stdout
  is not this process's text - it is whatever bytes an arbitrary program wrote
  to a pipe - and the package had already decided that for the same bytes
  elsewhere: the detached child's log file is read with `errors="replace"` (and
  says why), and the MuJoCo GL probe decodes both of its child's streams the
  same way. `sync_dataset_to_bucket`, which documents "Never raises on `hf`
  failure; errors are surfaced in the result dict" and which `stop_recording`
  calls unguarded on that promise, raised past a completed upload and took the
  episode/frame report with it; the VERA server's log pump died on the byte, so
  the lines its readiness error tells an operator to read stopped arriving.
  Every read of a child's stream now substitutes. A pipe this process writes
  stays strict, and the codec is unchanged.
