### Fixed: an IoT Thing named after a fleet topic can no longer publish the fleet stop

The robot policies grant every certificate `strands/<thing>/*`, and nothing
reserved the Thing names `safety` and `broadcast`, so a device registered as
`safety` could publish `strands/safety/estop` and `strands/safety/resume`
through its own key space, even under `strands-robot-no-estop`. Those names
(`RESERVED_THING_NAMES`, read from the policy documents) are now refused by
`provision_robot`, `provision_operator`, `reprovision_thing` and the Fleet
Provisioning hook, which also accepts only the serial or `<model>-<serial>` as
the Thing name (hook version 3, redeployed by `bootstrap_account`). The
no-estop policy denies `strands/safety/*` outright and the children policy
denies retained safety messages and the `broadcast` segment for every robot.
`reprovision_thing` no longer copies `strands-robot` silently: pass
`estop_publish=True` (CLI `--estop-publish keep`) or `False` (`drop`).
`withdraw_fleet_stop_grant` (CLI `withdraw-estop-publish`, dry run unless
`--apply`) moves an account's older certificates to the no-estop policy.
