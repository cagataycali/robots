### Docs: the six orphan dashboard pages went, and a grader pins the command list

`docs/dashboard/` launched an operator dashboard with `python -m strands_robots
dashboard --port 8090 --local-dev` on nine lines. That command does not exist:
`strands_robots.__main__` dispatches `doctor` and `verify-dataset` only, so the
documented first step exited 1. The pages also listed 54 HTTP endpoints, 5
websocket paths and 11 CLI flags -- one table saying it came from `--help` "on
this build" -- for a server the package never carried: `strands_robots/dashboard`
holds the modules an operator dashboard is built out of (WebAuthn auth, consent,
safety state, settings, log redaction, the agent motion gate) and no FastAPI app,
no routes and no argparse. All six pages were outside the MkDocs nav with no
inbound link, so nothing pointed at them either.

What only lived there is kept: the two agent-motion knobs
(`STRANDS_DASH_AGENT_PHYSICAL_MOTION`, `STRANDS_DASH_TASK_REQUIRES_CONFIRM`) move
to the README configuration table, and `doctor` -- previously invoked only in a
deleted orphan -- is documented in `docs/troubleshooting.md`.
`tests/test_docs_module_commands_are_dispatched.py` now fails on any page that
invokes a command `_COMMANDS` does not carry, or on a shipped command no page
invokes.
