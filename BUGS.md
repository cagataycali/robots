
## Q156 — a live module vouches for a dead export (2026-08-22, CLOSED in 4 iterations)

`scripts/check-lib-wired.mjs` proved every pure rule MODULE is imported from a screen, and passed while
`quietNotice()` — the rule the iteration existed to add — had no caller at all. A module reached by one
screen vouches for every export beside it, so a dead rule hides inside a live file: the same
"correct, tested, green, never ran" failure the guard was written to end, one level down.

Pass 2 now asks the question per EXPORT. Measured across 312 of them, and the first naive rule was wrong in a
way worth keeping: 79 look uncalled until test files count as callers, because this repo deliberately exports
tuning constants so a test asserts against the real number instead of a copy. Three classes, gating only on
the defect — test-only (by design), exported wider than used, and called by NOTHING anywhere.

Three were in that last class, each resolved differently, and none of them by guessing:

* **`endpoints.onBackendChange` — DELETED.** A subscription with no subscriber that could never have gained a
  useful one: its only trigger's caller reloads the page immediately after, and App remounts on
  `backendKey()`. A callback firing microseconds before the page dies is a decoy, not a rail. Deleting it
  exposed a second notifier (`setAuthToken`) within seconds. Replaced by a test that both halves of the
  identity move the remount key, so an invisible backend switch fails by name.
* **`jointHistory.heldSeconds` — WIRED.** Both joint labels claimed "last 60 seconds of movement" from the
  first frame, so a 3-second-old arm was announced as a minute of movement and its flat trace read as a
  minute of stillness. For a screen-reader user that label IS the chart. `historyClaim()` narrows the claim.
* **`cameraState.retryDelayMs` — DELETED, and its comment was FALSE.** Kept "only so the timing here stays
  comparable" with `planRetry`; it capped at 10s where the live `backoffMs` caps at 30s, a 3x divergence at
  the tail. A second source of truth kept for comparison is worse than none: the reader gets a wrong answer
  with a reassuring note attached. The ceiling is now asserted by name.

The guard paid for itself two iterations after landing: its stale-row check fired the instant `heldSeconds`
gained a caller, which is how the tolerance row got deleted instead of being left behind as a lie. The
TOLERATED map is empty, and an empty map is the goal state.

## Q155 — MEASURED CONTRADICTIONS on the live rig (2026-08-22 02:2xZ, read-only, cagatay away)

Evidence first, conclusions marked as such. Measured with the dashboard's own token against pid 2519
(alive 2d04h, terminal-blessed Aug 19 start), plus `lsof` and `ps`.

**(a) RETRACTED THE SAME NIGHT — THE PROBE WAS BROKEN, NOT THE PRODUCT.** ~~`presence.connected = True`
for an arm whose servo bus nobody holds.~~ `lsof` returns **zero lines for every pid in this context,
including the shell asking** (`lsof -p $$` → 0, `lsof -p 2519` → 0). It is not permitted to read fds here
at all, so its silence about the serial ports said nothing about the serial ports. Positive control, one
command, would have caught it before the entry was written. And `/api/devices/logs/so101-follower` confirms
the opposite of my claim: `hardware connected` at 13:58:52, immediately after the `hw_joints` state-probe
failure — the bus DID open, and the mute joints are the known Q26 sync-read collision, not a fictional
never-opened port. `connected` is `bool(inner.is_connected)` (mesh/core.py:1032) and is telling the truth.
LAW (second time this week a measurement, not the code, was the defect): **a probe used as evidence of
ABSENCE must first be run against a positive control.** A tool that cannot see anything reports exactly
what "nothing is there" reports. The earlier caretaker check that DID return byte offsets ran in a
different (terminal-blessed) context — so "lsof worked last time" is not transferable evidence either.
The original claim, kept struck through so nobody re-derives it:
~~
`/api/fleet` shows exactly two peers, `so101-follower` and `so101-leader`, both `stale=False`,
`connected=True`, `joints=0`, camera `main` present. But `lsof /dev/cu.usbmodem5AB01584281` and
`…5AB01818061` return **nothing at all** — no process holds either port. An earlier caretaker check
(2026-08-20) got holders WITH byte offsets from the same command, so this is a real change, not a tool
limitation. hardware_robot.py documents `is_connected` as deliberately describing THE MOTORS (blind
cameras are dropped so one refusing camera cannot report the whole robot as disconnected, lines 343-361).
A motors-truth flag reading True while no process holds the motor port is a contradiction.
CONSEQUENCE IF CONFIRMED: `statusSentence`/`connBadge` consume `hwConnected`, so the fleet asserts
"hardware connected" for two arms that never opened their bus. The mute-arm warning (jointAbsence) is
the only thing saving the screen from reading fully healthy.
~~ (end of the retracted claim.) DONE INSTEAD: the flag is set from `inner.is_connected` at every presence
publish, and the log proves the open succeeded, so there is nothing to fix in (a).

**(b) a peer disappeared from the fleet while its process kept running.**
The sim twin (`so101-follower-twin` + `…__so101`) is GONE from `/api/fleet` — yet pid 37603, the
dashboard-spawned SIM child (`sim.add_camera(name=f"{n}/front"…`), is alive with 1d01h44m uptime. So the
mesh pruned/lost a peer whose process the dashboard is still parenting. That is exactly the pair the U15
`origin` rail (managed vs external) and the ageing-protection share a source for, and it deserves a look:
a managed child should not be able to age out of the fleet while its process runs — or, if it can, the
managed list is the place that knows and should say so.

**(c) not a bug, recorded so the disk alarm is not re-raised:** root has **98Gi** available of 926Gi. The
Aug-21 bleed (32 -> 21 -> 18Gi) fully reversed on its own; nothing was deleted by any loop.

Neither (a) nor (b) was acted on: fixing them means respawning arms on hardware nobody is standing next
to, and the supervisor law is never to restart. Both are diagnosis-ready for the next iteration.

### 2026-08-22 — the two silent arms: THE DASHBOARD SIDE IS ALREADY FIXED, the running process is not

Measured today on the live server (`/api/fleet`): the fleet is down to **2 peers**, both real arms, both
`joints: 0` with `connected: true`, `hw: 'so_follower'`, and `joint_problem: null`. The sim twin and its
child are gone.

What the operator actually reads on those cards, quoted from the running page — this is working as
intended and needs no change: *"state is arriving, but carries no joint positions… the arm is alive and
talking; a safety lockout and a failed bus read both look like this"*, above an `IDLE?` caveat saying
stillness cannot be confirmed, so treat the arm as able to move.

The `leader` half of the cause is fixed in this tree and **proven against cagatay's real calibration
directory** today, not against a fixture — `robot_calibration_gap("so101", "leader")` returns:

> robot_id 'leader' has a calibration, but as a teleoperator:
> …/calibration/teleoperators/so101_leader/leader.json. A robot in real mode loads
> robots/so101_follower/leader.json, which does not exist, so the bus will refuse with 'has no
> calibration registered' and the arm will report presence with no joints. Calibrate this id as a robot,
> or spawn it with one that already is: follower, follower_arm, leader_arm

That is the diagnosis it took a caretaker sweep plus a dig through a 10-line child log ring buffer to
reach by hand on Aug 21, now produced from the filesystem at spawn time. `robot_id="follower"` and
`"follower_arm"` correctly produce no warning; an unknown id lists the ids that do exist.

**So there is nothing left to build here.** The pre-flight landed Aug 20, one day AFTER these arms were
spawned (13:58/13:59 on Aug 19), and `joint_problem` annotation postdates the running server too — which
is exactly why the live cards can only point at the log instead of naming the cause. Both surfaces come
alive the next time cagatay starts the dashboard from a terminal. No loop should respawn those arms:
they are hardware nobody is standing next to.
