### Quality: the control-loop throttle test pins the loop's wait, not the runner's speed

`TestPeriodIsTheOnlyThrottle` asserted the pathology by counting how many
actions an unthrottled loop applied inside a fixed 0.2 s window
(`assert applied > 1000`). Command throughput is a property of the machine, not
of the loop: on a loaded runner the same unthrottled loop applies 1-2 actions in
that window instead of ~2200, so the floor failed on the full suite while
passing when the file ran alone. Under load the count also stopped separating
the two regimes at all -- a throttled and an unthrottled loop both applied 2
actions -- so it carried no signal about the contract it was meant to pin.

Both regimes are now pinned on the delay the loop waits between two
`send_action` calls, recorded through a wrapper that delegates to the real
`asyncio.sleep`: a 50 Hz rate is waited as 20 ms after every command, and a
non-positive period is waited as nothing at all. That separation is exact at
every load, and it additionally pins the link the guard exists to protect --
that `control_frequency` reaches the servo bus as the inter-command wait.
