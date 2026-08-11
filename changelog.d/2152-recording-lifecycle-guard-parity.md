### Quality: drive the recording lifecycle guards on the Isaac and Newton capture paths

`IsaacSimulation` and `NewtonSimEngine` each define their own `start_recording` and
per-step capture hook in their backend mixin, so the guards around them are independent
copies of one contract rather than one shared implementation. Newton's lifecycle-guard
module pinned most of them and Isaac had no equivalent at all, leaving the resume branch,
the recorder-creation failure path and both hook early-outs undriven there; the early-out
that fires while `stop_recording` flushes the trailing episode - flag already False, the
recorder still attached - was driven on neither backend. Adds the Isaac module and
Newton's missing case, so a failed recorder creation is known to reset `recording`, an
existing dataset is known to resume rather than be recreated, and the hook is known to
return without writing a frame when there is no recorder or recording has already stopped.
