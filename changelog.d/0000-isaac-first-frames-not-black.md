### Fixed: Isaac camera frames are not black while the render product catches up

Right after a camera is created or the world is reset, and for a tick or two
during a rollout, an RTX render product hands back an empty buffer or one of
zeros. `get_observation` and `render` took that as the picture, so every
recording on Isaac began with black frames on some cameras: an so101 wrist
camera 0.26 m above a cube recorded its first two frames as all zeros (MuJoCo:
never), and a policy trained on the dataset saw black inputs at every episode
start. Rendering before `start_recording` did not help.

An empty or all-zero frame now gets up to six render-only ticks
(`SimulationApp.update()`, no physics step) before it is returned. A camera that
is still black after the whole budget really sees black; it is believed until it
next produces colour, so it pays the budget once, not on every frame.
