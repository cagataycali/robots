### Fixed: the G1 zero-torque shutdown frame survives a policy that overruns the stop budget

`_ControlLoop.stop` signals the control loop, joins its thread within a budget,
and returns whether the thread actually joined. `G1Driver.stop_task` reads that
value and reports `stopped=False` when the join times out - its docstring
explains why, and names the case: a caller-supplied policy that outlasts the
budget, "a remote inference call is the ordinary case". The two teardown paths
called the same method and dropped the answer.

For `cleanup` that was load-bearing rather than cosmetic. It halted the loop and
then closed `_pubs` and set it to `None` regardless of whether the halt had
taken, so on an overrun it removed the publisher from underneath a live 500 Hz
thread. `_emit_zero_torque` reads `self._driver._pubs` and returns silently when
it is `None`, so the zero-torque shutdown frame was never published: not by
`cleanup`, and not later when the policy finally returned and the loop reached
its own `finally`. That is the fall `cleanup`'s docstring says the path exists to
prevent, and the loop still recorded a clean caller-driven `exit_reason` of
`stop_task` on the way out.

`cleanup` now releases the publisher only once the loop is provably gone. A loop
that is still running keeps it, so the frame that loop publishes from its own
`finally` reaches the wire, and the overrun is reported at ERROR naming the
remedy - call `cleanup` again once the loop has exited. The subscribers close
either way, because the loop never reads them, and a second `cleanup` releases
the rest. `stop` carries no envelope and releases nothing, so its frame was
never at risk; it now logs the overrun rather than returning from shutdown in
silence while a thread still holds the wire.

The zero-torque contract already had a test, and it drives a policy that returns
immediately. Its join therefore always succeeded and the discarded value was
always `True`, so the fast path could not distinguish a checked halt from an
unchecked one. The new cells block a policy past the join budget, which is the
regime `stop_task` calls ordinary.
