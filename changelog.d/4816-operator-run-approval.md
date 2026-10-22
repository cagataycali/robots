### Changed: an agent's run on a real arm asks the operator once, not on every tick

An agent's `run()` on a real body asked the per-command `confirm=` on every tick (30 dialogs a second), so a
callable that always said yes was the only way to let a policy move. A real `Robot` now takes
`approve=callable(RunPlan) -> bool`, asked once with the whole plan (robot, policy, instruction, rate,
duration) before anything is acquired; without it the run is refused and nothing is written. The safety chain
(the e-stop, the joint range, the soft `safe` band, the interlocks and the validators) still runs on every
step, targets are no longer clipped to the limit first (an out-of-range target is refused, not silently
moved), and a target outside `safe` inside an approved run is refused instead of re-asked. `confirm=` stays
the per-command gate for `confirm_act` and the dashboard. Two stale-state fixes ride along: `Latch.release`
now clears `reason`/`engaged_at`/`engaged_wall`, and a robot's `manifest` is regenerated when a camera or twin
is added after it was first read (it was a `cached_property` that missed them).
