### Fixed: a caller-stopped G1/Go2 rollout names what stopped it

`get_task_status()` answers from a snapshot the control loop stashes on the
driver as it clears itself, which is the only thing a poller arriving after the
rollout's thread is gone can read. `_ControlLoop.stop()` recorded its reason
*after* `join()` returned, and the loop's `finally` stashes while that `join()`
is still blocked - so the stash predated the reason, and a rollout ended by a
caller reported `exit_reason=None`: a finished rollout with no cause, on the one
surface an operator has for "why did my robot stop moving".

The Go2 lost the most, because it is the driver that distinguishes *which*
caller stopped it. `stop_task()`, the mesh's `stop` verb and `cleanup()` each
pass their own word into the same loop, and all three collapsed into the same
absence - so a halted rollout, a shutdown and a teardown were indistinguishable
after the fact. `stop_task()`'s own return value did carry the reason, so the
same rollout answered two different things depending on when it was asked.

The reason is now recorded before the stop is signalled, which is what the UR
driver's loop already does (`self._finish("stopped")` on the stop path, before
the exit can be read). Precedence is unchanged: the record is first-writer-wins,
so a loop that already ended on its own budget keeps that more specific reason.
