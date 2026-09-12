### Fixed: a failed Device Connect bring-up releases the loop it ran on

`init_device_connect_sync` starts `init_device_connect` on a daemon thread and
parks that thread in `run_forever`, so the runtime it built can outlive the
call. The loop and the thread reach the caller one way only: they are adopted
onto the returned runtime. A bring-up that raised has no runtime, so both were
unreachable - and the thread parked anyway, keeping an idle event loop, and the
epoll and self-pipe descriptors it holds, alive for the life of the process,
once per failed attempt. Six retries of an unreachable broker left six parked
threads, six open loops and eighteen held descriptors, none of them reachable
from the caller that was handed the failure.

The bring-up thread now closes its own loop and returns when the bring-up
failed - on the thread that owns the loop, where the close cannot race a runner
- and the wrapper waits `_LOOP_JOIN_TIMEOUT_S` for it before raising, so the
failure the caller receives also means the machinery is gone. A thread that
outlasts that budget is logged and keeps its loop: closing a running loop
raises, and reporting a release that did not happen is worse than one open
loop. The expired-budget outcome is unchanged - that bring-up is still running,
so there is nothing there to release.
