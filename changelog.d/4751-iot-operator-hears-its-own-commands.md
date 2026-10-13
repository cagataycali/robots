### Fixed: an operator on the shipped IoT policies can start a mesh

`Mesh.start()` subscribes every peer to its own `cmd` topic and to
`strands/broadcast`, operators included, but the operator policies granted
neither. AWS IoT Core answered the subscribe with reason code 135 and closed the
whole MQTT session, so an operator's mesh never came alive and the dashboard's
safety rail showed "safety mesh unavailable". `strands-operator-observe` now
grants Subscribe and Receive on the operator's own `cmd` and on `broadcast`
(another peer's `cmd` stays unreadable), and `reprovision_thing` republishes
that policy too. A refused subscribe with code 135 now names the missing
`iot:Subscribe` grant and the command that republishes the policies.
