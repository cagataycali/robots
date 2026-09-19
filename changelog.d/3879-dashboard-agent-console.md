### Added: the Agent tab - a Strands Agent whose hands are the simulations

`/ws/agent` is one operator conversation with a Strands `Agent`
(`dashboard/agent_console.py`, model from `STRANDS_MODEL_ID`, else the SDK's own default). Its tools
(`robots`, `sim_sessions`, `sim_start`, `sim_state`, `sim_set_joints`,
`sim_reset`, `sim_stop`, `emergency_stop`) go through the same `Safety`
object as the buttons, so the e-stop refuses the agent the way it refuses a
click and stopping is never refused. `sim_set_joints` raises the SDK
interrupt (`sim_motion`) before it runs; the browser shows a consent card
with exactly what a yes moves (`2 → 1.000 rad`), and *Allow once*, *Allow
for this conversation* or *Refuse* resumes the same turn. Grants live in the
socket and die with it; every answer is written to the HITL audit log.
