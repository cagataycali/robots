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

### Removed
- The per-unit teleop input envelope this branch had added to
  `mesh/security.py` (`INPUT_ENVELOPES_BY_UNIT`, `NORM_MODE_UNITS`,
  `input_envelope_for_units`) is dropped: nothing called it, nothing tested
  it, its `rad` row was byte-identical to the `deg` row it existed to differ
  from, and the comment justifying it described `DEFAULT_INPUT_VALUE_ABS` as
  a `4*pi` radian assumption when the constant has been `720.0` frame units
  on this tree throughout. Unwired safety machinery reads as a solved problem
  to the next reader, so it leaves with AGENTS.md convention 10. The
  `value_abs_by_key` / `max_slew_by_key` parameters stay as the extension
  seam and now say plainly that nothing populates them yet. Wiring it to the
  receiving robot's declared `norm_mode` is tracked in #2935, where it can
  carry its own mixed-unit tests instead of riding a dashboard change.
