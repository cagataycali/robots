### Fixed: a Reachy daemon body that is not a JSON object no longer crashes the driver

`reachy_transport.api` hands the decoded body back unreshaped - `json.loads`
decodes any JSON value - and the native driver required an object at six sites
while judging it at none. A daemon (or an interposed proxy) answering with an
array, a string, a number, `null` or `true` reached `result.get("error")` and
raised `AttributeError: 'list' object has no attribute 'get'` out of
`connect_eagerly`, `wake_up`, `goto_sleep`, `stop_task`, `play_move` and `stop` -
methods documented to report a reason and leave the driver usable. `list_moves`
had the mirror defect: it graded dict-vs-not rather than array-or-not, so a
scalar body was returned as `{"status": "success", "moves": "ok"}`. Each door now
refuses the shape it cannot read, naming the call, the JSON type and a preview of
the body, and `api` is typed `Any` so a caller that needs an object judges it -
the rule `ReachyMiniDriver._transport_failure` already stated for this transport.
The transport's own `{"error": ...}` failures reach the caller unchanged.
