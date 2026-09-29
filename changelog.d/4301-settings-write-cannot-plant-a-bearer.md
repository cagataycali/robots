### Fixed: a dashboard settings write cannot plant the standing bearer

`POST /api/config` and `POST /api/settings` refuse `security.auth_token`, the
bearer every `/api` and `/ws` request may present. Before, any admitted session,
the pre enrolment loopback posture included, could write one to `settings.json`,
and it kept admitting its holder after every passkey was deleted. The env
spelling of the same value, `DASHBOARD_AUTH_TOKEN`, was already refused on the
`env` half of the request; the two spellings now share one fence,
`config_api.REFUSED_SETTINGS_KEYS`. A bearer is set on the host through
`DASHBOARD_AUTH_TOKEN`; clearing one from the page still works. (f020)
