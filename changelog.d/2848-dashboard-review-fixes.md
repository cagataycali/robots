### Fixed
- Record-panel docs stopped citing "`/api/record` does not exist yet": the
  router is mounted at this head (`strands_robots/dashboard/record_api.py`
  and `server.py`), so quickstart, collect-train-deploy and troubleshooting
  now describe the live behaviour and treat the mock as a wiring problem.
- `tests/test_dashboard_calibration.py::test_saved_wins_over_everything_before_it`
  no longer embeds `/Users/x/...` in a fixture string; the `no-host-paths`
  regex sweep was catching it in `tests/`, turning a required check red on
  every dashboard PR. `~/.cache/...` reads the same to the parser.
- Dashboard `peer_tools.py` imports `SIM_CALL_BLOCKED_ACTIONS` from
  `mesh.security` instead of literal-duplicating it, and drops the em-dash
  from a tool-parameter description the model reads back verbatim
  (project rule: ASCII in runtime strings).
