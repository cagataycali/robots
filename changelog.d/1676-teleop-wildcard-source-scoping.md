### Fixed: a teleop receiver refuses a wildcard source instead of following every peer

`InputPublisher` / `InputReceiver` interpolate `device_name` and
`source_peer_id` into the Zenoh key expression
`strands/{peer_id}/input/{device_name}`, and Zenoh reads `*` / `**` as
wildcards. Nothing validated either identifier where the stream is built, so
source scoping - the only thing making teleop point-to-point - could be
switched off by a single argument:

```python
rx = InputReceiver(mesh, robot, source_peer_id="**")   # returned a receiver
rx.topic          # 'strands/**/input/leader'
# every peer publishing an input frame now reaches robot.send_action()
```

The wire `teleop_receive` command already rejected exactly those values, so the
remote surface was stricter than the local API it delegates to
(`HardwareRobot.start_teleop_receive` / `start_teleop_publish`, which accepted
them too). The shared rule is now one validator,
`strands_robots.mesh.security.validate_mesh_identifier`, called from
`validate_command` and from both constructors; the two `HardwareRobot` entry
points report through the tool envelope and validate before tearing down an
existing stream, so a rejected call cannot stop a live one. Well-formed
identifiers and their key expressions are unchanged.
