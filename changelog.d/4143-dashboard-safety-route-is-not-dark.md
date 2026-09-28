### Fixed: the dashboard no longer reports its own e-stop route as a dark feature

The Sim tab posted to a templated `/api/safety/${action}`, so the bundle's route
list carried `/api/safety/{p}` while the server publishes only the literals
`/api/safety/estop` and `/api/safety/resume`. Every page load on a matching
build showed "Older server. 1 feature on this page is dark". The call now names
the two literal routes, and a test grades every route the shipped bundle calls
against the server's own `openapi.json`.
