### Fixed: a robot's child peers share the Thing's IoT key space; AWS IoT no longer drops the session on every child heartbeat

A MuJoCo `Robot("so101")` on `STRANDS_MESH_BACKEND=iot` (or `bridge`) attaches
its robot as a child peer `<thing>__so101` that publishes presence, state and
cameras over the Thing's own MQTT session. The `strands-robot` and
`strands-robot-no-estop` policies granted `strands/<thing>/*` alone, and AWS
IoT answers an ungranted publish by ending the session: every real robot on
the iot backend lived in a connect/disconnect cycle of about 150 ms, its
presence reached the fleet once in 30 s, and nothing above DEBUG said why.

- A new policy, `strands-robot-children`, grants
  `strands/${iot:Connection.Thing.ThingName}__*/*` in every statement a child
  needs (publish, reply, direct reply, subscribe, receive on `cmd` and
  `response`). `provision_robot` and the Fleet Provisioning template attach it
  to every robot certificate next to `strands-robot` or
  `strands-robot-no-estop`; those two documents are unchanged (an AWS IoT
  policy document is capped at 2,048 characters and they have no room, which
  AWS refused live). Verified live: 574 publishes on
  `strands/childfix-a__so101/state` in 60 s with zero disconnects, while
  `strands/childfix-b/state` and `strands/childfix-ax/state` from the same
  certificate each ended the session (reason code 135).
- A Thing name may not contain `__`, the child separator, so a second Thing can
  never sit inside another's key space (`provision_robot`, `provision_operator`
  and `reprovision_thing` refuse it before any AWS call).
- `IotMqttTransport` WARNs when the broker ends the session within a second
  of a publish, naming every topic published inside that second (newest
  first; the broker's DISCONNECT lands 47 to 74 ms after the offending publish,
  after a 10 Hz state loop has published again) and the reprovision command,
  once per set of topics.
- The presence shadow mirror is wired for the Thing only. A child peer is not a
  Thing and `$aws/things/<thing>__so101/shadow/...` is granted by no policy, so
  the child's shadow update was ending the session every 1.5 s even with the
  child key space granted.
- `strands-robots doctor` has an `IoT Child Peers` row that reads the Thing's
  attached policy from the control plane and fails with the fix when the grant
  is missing.
- `reprovision_thing` (`strands-robots iot reprovision <thing>`) attaches
  `strands-robot-children` to a robot certificate that predates it and
  republishes the module-owned policies when their documents changed.

**Existing fleets:** run `strands-robots iot reprovision <thing>` (or re-run
`provision_robot(<thing>)`) for each robot Thing, then restart the robot; a
Fleet Provisioning account re-runs `bootstrap_account()` so the template
attaches the new policy to future devices. Nothing changes for the operator
policy: `strands/+/state` already matched `strands/<thing>__so101/state`.

The blame WARNING stays quiet for a disconnect this process asked for:
`close()` raises a closing flag before stopping the client, since awscrt
reports the stop through the same lifecycle callback as a broker DISCONNECT
and every 10 Hz publisher has a publish inside the window at shutdown. Without
it each normal exit told the owner to reprovision a healthy Thing.
