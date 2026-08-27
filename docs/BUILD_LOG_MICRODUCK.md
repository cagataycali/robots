# BUILD LOG — microduck Layer B/C/agent (this lane)

- 2026-08-27: pushed inherited HEAD 2eb39829 (agent example) to fork/microduck-e2e.
- 2026-08-27: Layer B piece 1 — drivers/microduck.py wire codec + 15→14 joint map. NDJSON JSON-RPC frames proven byte-exact vs duck-ipc-proto Rust contract (robot.move notification, robot.do request snake_case skill, hello api_version=16, empty-params unit calls). action_to_wire fixed-order twist/head/pose/mouth/skill with skill validation. map_hardware_joints drops mouth@idx9.
