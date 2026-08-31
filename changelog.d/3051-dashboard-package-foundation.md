### Added: `strands_robots.dashboard` package foundation and the `[dashboard]` extra

The operator dashboard package opens with the five modules that import no
dashboard sibling, so each is usable on its own rather than only once the whole
package is present: `auth` (WebAuthn ceremonies, credential store, challenge
caps), `settings` (resolved configuration and override isolation),
`log_redaction` (secret registration and log scrubbing), `task_timeout` (ack
budget and timeout verdicts) and `ttl_cache` (a bounded time-to-live cache).

`strands_robots/dashboard/__init__.py` is the single optional-dependency
chokepoint. Python executes a package `__init__` before any of its submodules,
so one `require_optionals` call there covers every module the package will ever
gain, and the refusal names `strands-robots[dashboard]` -- the extra that
actually supplies the missing module -- rather than letting a bare
`ModuleNotFoundError: No module named 'fastapi'` read as a broken virtualenv and
send the reader hunting in their own environment. `require_optionals` (plural) is
deliberate: the absent extra means several modules are missing at once, and
reporting them one `ImportError` at a time turns one install into three round
trips.

The `[dashboard]` extra declares `fastapi`, `uvicorn`, `webauthn` and
`python-multipart`, and is named in `[all]`. `[tool.hatch.build.targets.wheel]`
packages `strands_robots` entire, so these modules ship in every install while
those four dependencies do not -- the extra is what makes them reachable.
`python-multipart` is in the extra but absent from the refusal list: it is a
runtime dependency of FastAPI's form parsing rather than something this package
imports, so it has no module name worth naming there.

No public entry point is added yet. A `main()` on the package would reach `cli`,
which reaches `server`, and neither lands here -- an exported callable that
raises `ImportError` is worse than an absent one, so it arrives with the modules
that supply it.
