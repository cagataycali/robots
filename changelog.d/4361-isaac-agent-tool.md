### Added: the Isaac simulation is a Strands AgentTool, and `run_agent` drives an agent on it

`Robot("so101", backend="isaac")` is presented as a tool an agent drives, but
`IsaacSimulation` had no `tool_spec` and no `stream`: `Agent(tools=[sim])`
logged "unrecognized tool specification" and the agent got no tools at all.

`IsaacSimulation` is now an `AgentTool`. Its `tool_spec` publishes the shared
simulation schema with the `action` enum narrowed to the actions this backend
implements (43 of the 77), and names each robot's joints; `stream` and
`sim(action=...)` route an action to the method of that name, forwarding only
the arguments it takes and saying which were ignored. Kit only advances on the
thread that created `SimulationApp`, while Strands runs tools on a worker, so a
tool call is marshalled onto the main-thread pump, and without one it is
refused with the recipe instead of blocking. `sim.run_agent(agent, prompt)` is
that recipe: it runs the agent on a worker while the calling thread pumps.

The pump also picks up a worker's call within 5 ms instead of after a fixed
50 ms sleep plus a forced re-render: a marshalled `send_action` from a worker
went from 75 ms to 7.2 ms (13 Hz to 139 Hz) on one L40S.
