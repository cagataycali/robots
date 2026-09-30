### Added: a Reset button on every dashboard robot card

`POST /api/robots/{peer_id}/reset` sends the peer's wire `reset` through the
same gate as any other command. `reset` joins
`strands_robots.dashboard.agent_motion.GATED_ACTIONS`: a real arm's reset drives
every joint to the home pose at once, so it needs the browser's confirmation
(`confirmed: true`, a string is refused) or the operator's grant, and the
refusal is the gate's own sentence with nothing sent. A simulated child peer is
routed to its parent world, which resets every robot it holds. A peer whose
presence reports a task connecting or running is refused with 409 before the
RPC: stop it first. The card's run form gains the button next to stop and run,
disabled while a task runs, a request is in flight or the card is stale, with
the reason as its title; a real arm opens the confirm sheet first. The outcome
line shows the server's sentence, never a guess.
