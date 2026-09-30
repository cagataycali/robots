### Fixed: a retained command is refused, not executed at every boot; the operator policy can no longer store one

MQTT replays a retained message to every new subscription, `Mesh.start()`
subscribes `cmd` and `broadcast` at every boot, and the replay cache is per
process, so one `--retain` publish on `strands/<thing>/cmd` by any operator
certificate was a motion-on-boot implant: live, a stored `execute` ran a
rollout at the robot's next start with nobody present, and a stored
`set_joints` was dispatched on three starts out of three. `Mesh._on_cmd` now
refuses a sample whose RETAIN flag is set, audits `command_refused` with
`reason=retained`, and WARNs once per topic with the command that clears it.
`OperatorPublishToFleet` is `iot:Publish` only on `strands/*/cmd` and
`strands/broadcast`; presence and health keep `RetainPublish`.
