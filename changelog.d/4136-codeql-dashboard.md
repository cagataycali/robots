### Fixed: the 23 CodeQL alerts under `strands_robots/dashboard/` and the 8 frontend advisories

Every client-named path the dashboard inspects (`routes_record.contained_path`,
`training.contain`, the record screen's dataset directory) is folded with
`os.path.realpath`/`normpath` and has to be the home or start with it before
anything on disk is consulted; the acceptance and every refusal sentence are
unchanged, the barrier is one CodeQL recognises. Caller-supplied peer ids,
camera names and audit targets pass through `log_redaction.one_line` before
they reach a log line. Session details and the job-ledger problem sentence
name an exception's type instead of quoting it; the words go to the server log.
The frontend moves to vite 6.4.3 (esbuild 0.25) and its lockfile drops the
stale `vite-plugin-pwa` tree where `fast-uri` lived; the bundle is rebuilt.
