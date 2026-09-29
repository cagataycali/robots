### Fixed: a policy running on a simulation peer shows as running (#4182)

A sim peer publishes its rollout as `state.robots.<name>.active` and never
`task.status`, so the card's dot stayed green and `fleet peers` said `idle`
while the joints moved. The card, the status sentence and the voice agent's
fleet listing now read that flag when no task status is present; a child peer
(`<sim>__<robot>`) is asked about its own robot only.
