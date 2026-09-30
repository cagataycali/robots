### Fixed: a robot's child peers share the Thing's IoT key space; AWS IoT no longer drops the session on every child heartbeat

A MuJoCo `Robot("so101")` on `STRANDS_MESH_BACKEND=iot` (or `bridge`) attaches
its robot as a child peer `<thing>__so101` that publishes presence, state and
cameras over the Thing's own MQTT session. The `strands-robot` and
`strands-robot-no-estop` policies granted `strands/<thing>/*` alone, and AWS
IoT answers an ungranted publish by ending the session: every real robot on
the iot backend lived in a connect/disconnect cycle of about 150 ms, its
presence reached the fleet once in 30 s, and nothing above DEBUG said why.

- The robot policies now grant `strands/${iot:Connection.Thing.ThingName}__*/*`
  next to the Thing's own topics in every statement a child needs (publish,
  reply, direct reply, subscribe, receive on `cmd` and `response`). Verified
  live: 574 publishes on `strands/childfix-a__so101/state` in 60 s with zero
  disconnects, while `strands/childfix-b/state` and `strands/childfix-ax/state`
  from the same certificate each ended the session (reason code 135).
- A Thing name may not contain `__`, the child separator, so a second Thing can
  never sit inside another's key space (`provision_robot`, `provision_operator`
  and `reprovision_thing` refuse it before any AWS call).
- `IotMqttTransport` WARNs once per topic when the broker ends the session
  within a second of a publish, naming the topic and the command.
- `strands-robots doctor` has an `IoT Child Peers` row that reads the Thing's
  attached policy from the control plane and fails with the fix when the grant
  is missing.
- `reprovision_thing` (`strands-robots iot reprovision <thing>`) republishes
  the module-owned policies as the new default version.

**Existing fleets:** run `strands-robots iot reprovision <thing>` once for any
Thing (the policy is shared, every certificate it is attached to picks the grant
up at its next connect), or re-run `provision_robot(<thing>)`, then restart the
robots. Nothing changes for the operator policy: `strands/+/state` already
matched `strands/<thing>__so101/state`.
