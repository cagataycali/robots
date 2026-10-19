### Fixed: a refused run leaves no policy session attached

`Robot.run` and `Simulation.run` call a policy's `detach()` on every exit path once its `attach()` has run.
A failure after `attach` (a wrong-type `hz`, unit resolution, a `requirements()` refusal, `reset()`, or a
`run.started` listener) used to leave the session open, because the run's `try` began after the bind. A named
units scheme (`units="lerobot"`, or an unknown one) is now checked before `attach`, so a refusal there opens
nothing. In lock-step, a partner refused in its bind is detached along with the robots already bound.
