### Fixed: `G1Driver.run_policy` / `stop_task` / `get_task_status` now wire the 500 Hz control loop

Previously the three verbs returned `"not wired yet (issue #358)"` refusal
envelopes. The transport primitive the loop needs is `send_action` (landed in
issue #361 PR-B), so the loop can be wired without the `g1_tools` motion
verbs from #358 - the two concerns stay separable.

`run_policy(policy_object, duration=..., n_steps=...)` spawns a daemon thread
that runs at 500 Hz (matching the SDK's own G1 example cadence), builds a
`LowCmd_` from the policy's per-step return, and publishes it on `rt/lowcmd`
through the same `DDSPublisher` `send_action` uses. Every step re-gates
through `_check_motion_gates("motion")` so an FSM transition out of
`HANDSHAKE_FSMS ∪ WALK_FSMS` refuses the *step* rather than the whole task -
the loop exits, publishes a zero-torque `LowCmd_` (from
`_build_zero_torque_lowcmd`, a soft *controlled* stop with Enable on the
named joints), and drops its reference. Same posture on every terminal path
except `publish` (where the wire has just refused; a second stamp would
clobber the reason with a fresh error).

Every exit path names itself in `get_task_status`: `stop_task`, `n_steps`,
`duration`, `gate` (with the refusal text), `policy` (with the exception or
the reason the action was unusable), `publish`.

`duration` and `n_steps` are validated up front on the shared
`positive_finite_number_error` / `positive_count_error` domains
`HardwareRobot` already refuses on, so the same budget cannot be refused for
a digital twin and accepted for the biped it mirrors. `duration=nan` /
`inf` / `0` / negative and `n_steps=True` / `0` / negative / fractional
return refusal envelopes rather than reaching the loop.

Concurrent `run_policy` calls admit through `_task_admission`, a
`threading.Lock` mirroring `HardwareRobot._task_admission`. The lock is
held across the `is_running` check, the `self._loop` assignment and
`start()`, so two threads cannot both start a rollout on `rt/lowcmd` and
an e-stop landing between the check and the start cannot count this peer
as stopped while the rollout starts a moment later.

`G1Driver.cleanup()` and `stop()` join the running loop before closing
`_pubs`, so the zero-torque shutdown frame goes out on a publisher that
still exists. The pre-review head closed `_pubs` under the live 500 Hz
thread, which dropped the loop into its `publish` branch and skipped the
zero-torque frame - the fall the whole path exists to prevent.

`start_task` still refuses because the provider registry that turns an
`instruction: str + policy_provider: "groot"` into a policy object lives in
`strands_robots.policies` and threads through the `g1_tools` motion verbs
(#358). The refusal names both facts and points at `run_policy` as the
already-live path.

Contract-graded off the driver: `tests/drivers/test_g1_control_loop.py`
exercises every exit branch with a callable-double policy and a recording
publisher; the `unitree_sdk2py` submodules are stubbed via
`monkeypatch.setitem(sys.modules, ...)` in an autouse fixture, so the
production publish lane (`_build_lowcmd_from_action` → `LowCmd_`) is
graded on an SDK-less CI box rather than an SDK-less fallback branch
hardware can never take.

### Round 4 - terminal snapshot survives loop teardown, stop_task reports the join, in-flight action is dropped on stop

Round 3's review found three shapes the earlier heads left open. They land
here because each is a caller-visible contract, not a rewrite of the loop.

**`get_task_status` no longer collapses five of six exit reasons.**
`_run`'s `finally` used to clear `self._driver._loop`, and `get_task_status`
read that reference, so once the loop ended by itself every self-terminating
exit path (`n_steps`, `duration`, `gate`, `policy`, `publish`) round-tripped
to the caller as `"no task has been started on this driver"`. The loop now
stashes its terminal `snapshot()` in `_last_task_snapshot` under the
admission lock *before* clearing `_loop`, and `get_task_status` returns the
stashed value when the live loop is gone. `run_policy` clears the stash on
admission so a poller between two rollouts cannot read the previous rollout's
exit. Six new test cells grade the six exit reasons individually plus the
cross-rollout stash-clear.

**`stop_task` reports the join outcome honestly.** `_ControlLoop.stop` now
returns whether the thread joined within its timeout (`thread.is_alive()`
after the join), and `stop_task` surfaces that as `stopped=True/False` in the
payload and as `status="error"` when the join failed. The previous shape
returned `status="success"` while the payload's own `running=True` said the
loop was still writing frames - a state the caller could not read as a
"stopped" signal without contradicting itself. A blocking-policy test case
grades this by tripping a `threading.Event`-gated policy against a shortened
join timeout.

**A stop signal between the policy call and the publish drops the pending
frame.** The loop only read `_stop_event` at the top of each iteration, so a
stop signal arriving *while* the policy was computing (a remote inference
call is the ordinary case) was only observed on the next pass - but by then
the in-flight action had already reached `rt/lowcmd`, and only *then* did the
zero-torque frame go out. The loop now re-reads `_stop_event.is_set()` after
`_call_policy` returns and before the publish; the finally still stamps the
zero-torque frame, so the wire's last frame is the stop frame rather than a
fresh position command followed by the stop frame. Graded by a recording
publisher whose last frame's enabled-slot count separates action (1 enabled)
from zero-torque (29 enabled).

The `_fsm_id` producer the round-3 review names is `#2765`'s scope
(motion-switcher API wiring) and stays there; the acceptance item this PR
owns is the loop's control-flow, which is what round 4 tightens.
