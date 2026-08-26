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
the reason the action was unusable), `publish`. A loop that ended by itself
still reports the reason it ended: the loop stashes its terminal `snapshot()`
under the admission lock before dropping the driver's reference to it, and
`get_task_status` reads that stash once the live loop is gone, so the five
self-terminating reasons (`n_steps`, `duration`, `gate`, `policy`, `publish`)
are distinguishable from a driver on which nothing was ever started.
`run_policy` clears the stash when it admits, so a poller between two
rollouts cannot read the previous rollout's exit.

`stop_task` reports the join outcome rather than assuming it: it surfaces
whether the loop's thread actually joined within the timeout as `stopped`
in the payload, and returns `status="error"` when it did not, so a caller
never reads `"success"` beside a payload whose own `running` flag says
frames are still going out.

A stop signal arriving *while* the policy is computing - the ordinary case
for a remote inference call - drops the action it was computing. The loop
re-reads the stop event after the policy returns and before the publish, so
the last frame on the wire is the zero-torque stop frame rather than a fresh
position command followed by it.

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
still exists. Closing `_pubs` under the live 500 Hz thread dropped the loop
into its `publish` branch and skipped the zero-torque frame - the fall the
whole path exists to prevent.

`start_task` still refuses because the provider registry that turns an
`instruction: str + policy_provider: "groot"` into a policy object lives in
`strands_robots.policies` and threads through the `g1_tools` motion verbs
(#358). The refusal names both facts and points at `run_policy` as the
already-live path. The `_fsm_id` producer the per-step gate reads is
#2765's scope (motion-switcher API wiring) and stays there.

Contract-graded off the driver: `tests/drivers/test_g1_control_loop.py`
exercises every exit branch with a callable-double policy and a recording
publisher; the `unitree_sdk2py` submodules are stubbed via
`monkeypatch.setitem(sys.modules, ...)` in an autouse fixture, so the
production publish lane (`_build_lowcmd_from_action` → `LowCmd_`) is
graded on an SDK-less CI box rather than an SDK-less fallback branch
hardware can never take.
