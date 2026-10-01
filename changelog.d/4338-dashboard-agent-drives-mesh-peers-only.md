### Changed: the dashboard is the fleet on the mesh, its cards, and the agent that drives them

The dashboard agent console drives mesh peers only. Its seven in-process
simulation tools (`robots`, `sim_sessions`, `sim_start`, `sim_state`,
`sim_set_joints`, `sim_reset`, `sim_stop`) and the `MotionGate` that gated
only `sim_set_joints` are gone from `strands_robots.dashboard.agent_console`.
What stays: `fleet`, `spawn_robot`, `despawn_robot`, one proxy tool per
tool-worthy peer, and `emergency_stop`. A robot the agent moves is a mesh peer,
gated by `MotionInterruptHook`, or it is not moved from here.

The agent's `emergency_stop` now stops the fleet the way the red button does:
`agent_console.fleet_stop` latches the dashboard's own lockout, asks every live
peer to `stop`, and fires the signed fleet e-stop, the same two rails
`POST /api/mesh/safety/estop` fires. `all_stopped` is true only when every live
peer confirmed. Stopping is never gated.

The record, train and sim screens leave the page: the router keeps settings,
activity, devices, e-stop and help; the `#record`, `#training` and `#sim`
fragments fall back to the fleet. The backend routes those screens called
(`/api/record`, `/api/collect`, `/api/replay`, `/api/training/*`, `/api/sim/*`)
stay for API users this round; only the page stopped calling them.

`GET /api/agent` reports `asks_first` derived from the mesh (the real-arm
peers' proxy tool names) and the fleet hook's interrupt name,
`physical_motion`. Breaking for `/ws/agent` clients that rendered the retired
`sim_motion` interrupt's `reason.session_id`: the fleet hook's `reason.target`
is what arrives now.
