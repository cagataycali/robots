### Fixed: Isaac cameras show the state the simulation is in, a recorded episode is the rollout that ran, and an episode no longer opens on a black wrist view

Isaac's RTX camera products - every camera but the first - refresh only on a Kit app
update that advances the timeline; a render-only `World.render()` never lights them.
So a multi-camera observation ran an extra `SimulationApp.update()` that integrated a
whole `rendering_dt` (four physics steps) `sim_time` never counted: on Isaac Sim 6.1,
150 recorded steps at 15 Hz integrated 15.03 s of physics where the same rollout
unrecorded ran 10.0 s, ending on another pose, and dataset timestamps understated the
physics 1.5x. And after `reset()` the products held no frame of the reset scene for
~6 updates, so the first observation of every episode showed the policy a black
wrist view. The World is now built with `rendering_dt == physics_dt`, so a rendering
`step()` / `send_action()` tick is ONE physics step that also refreshes every camera,
and `get_observation` only ticks the renderer when the last physics tick did not
render. `reset()` renders up to 12 physics ticks until every camera returns a lit
frame, then zeroes every robot joint and object velocity before the clock is rewound.
Recorded and unrecorded rollouts now integrate the same 10.0 s and end on the same
joints, and the first observation after a reset has every camera lit.
`IsaacConfig.rendering_dt` is accepted for compatibility and no longer changes the
physics-render coupling.
