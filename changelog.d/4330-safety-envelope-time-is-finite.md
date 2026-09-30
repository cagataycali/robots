### Fixed: a safety envelope with a non-finite `t` no longer breaks the fleet view

One e-stop published with `"t": NaN` (a token `json.loads` accepts) gave the
dashboard's lockout `since=nan`, every card repeated it, and `/ws/mesh` wrote
it as a bare `NaN` the browser's `JSON.parse` refuses, so the fleet view never
loaded again until a finite `t` arrived or the dashboard restarted. The
dashboard now reads the peers' own rule (`security.as_wire_timestamp`): a
present but non-finite `t` drops the envelope with a log line and a refused
activity row, and the `/ws/mesh` boundary serialises with `allow_nan=False`,
sending any non-finite float as `null`.
