### Fixed: `PolicyServer` names the `[inference]` extra when `websockets` is missing

On an install without `websockets` (or with one older than 17.1),
`PolicyServer.start()` and `.serve()` failed on a bare
`ModuleNotFoundError: No module named 'websockets'`, after the policy had
already been built. The server now refuses at construction, before any
checkpoint is loaded, with the report `RemotePolicy` gives:
`pip install 'strands-robots[inference]'`.
