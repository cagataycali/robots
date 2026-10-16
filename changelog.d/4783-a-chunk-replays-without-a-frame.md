### Changed: the run loop renders a camera only on a step the policy looks at

`Robot.run` and `Simulation.run` used to render every camera on every step, though a `ChunkedPolicy` reads an
observation only when it asks its inner policy for a new chunk. A policy now answers
`needs_observation(instruction=)` (the `Policy` default is `True`, every step); `ChunkedPolicy` answers `True` only
on a replanning step, so a chunk of 10 renders once per 10 steps. A run with `stop_when=` still sees a frame on every
step, and `bind` still reads one frame for the requirements check. A robot on a bus still reads its
cameras every step: the skip is a render saving, and a hardware camera left undrained hands back a stale frame.
