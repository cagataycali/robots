### Added: Pollen Microduck native driver — `Robot("microduck", mode="real")` over robotd

The Microduck's 14-DOF biped is now driven natively. `MicroduckDriver` speaks
the on-robot `robotd` daemon's `duck-ipc-proto` JSON-RPC directly over its Unix
socket (forward it over SSH for a remote robot), so the *same* code that runs a
policy in `mode="sim"` drives the physical robot in `mode="real"`. A Hello
handshake pins the API version (a mismatch refuses rather than mis-parses), a
`robot.subscribe` stream feeds joints/pose/battery to the mesh, and action dicts
map onto `robot.move`/`head`/`pose` intents and `robot.do` skills. On-device
policy ownership is honoured: `run_policy`/`start_task` refuse and point at the
intent path, because `robotd` runs the walking/skill ONNX itself and exposes no
per-joint write. `robots.json` gains `hardware.driver="strands"`, so the driver
resolves by name with no `driver=` guess needed.
