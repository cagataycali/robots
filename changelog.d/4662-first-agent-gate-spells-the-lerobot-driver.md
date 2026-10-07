### Fixed: the first-agent operator-gate example registers its robot as a tool again

`docs/start/first-agent.md`'s "gate in front of real motion" example built
`Robot("so101", mode="real", port="/dev/null")`. Since the SO-101 resolves to
its native `FeetechDriver` by default, and that driver is not a Strands
`AgentTool`, `Agent(tools=[arm])` logged "unrecognized tool specification" and
registered nothing, so the `robot-command-approval` interrupt the page teaches
was never reached. The example now spells `driver="lerobot"` and the paragraph
says why. A docs test grades every fence that hands a `mode="real"` robot to
`Agent(tools=[...])` and refuses one that builds a driver Strands drops.
