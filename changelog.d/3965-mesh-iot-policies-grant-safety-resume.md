### Fixed: the IoT policies grant `strands/safety/resume` beside `strands/safety/estop`, so a cloud-delivered fleet lockout can be cleared

`Mesh.start` subscribes `safety/estop` and `safety/resume` as a pair, and the
bridge forwards both to MQTT by default (`DEFAULT_BRIDGE_SUFFIXES`). The
provisioned `strands-robot` policy granted Subscribe + Receive on estop only, so
every robot engaged a lockout delivered over IoT Core and the broker then
denied the one topic that lifts it: the fleet stayed locked until each robot
was restarted by hand. The `strands-operator` policy could publish the stop but
not the release, and could observe the stop but never the release.

Robot policies now Subscribe + Receive `safety/resume`; the designated
safety-authority robot policy publishes it alongside estop under the same
`allow_estop_publish` gate, so a cert that may not originate a stop cannot
clear one either (a Last Will armed on `safety/resume` would be the estop
dead-man switch pointed the other way). The operator policy publishes and
observes both halves. Authorising the delivery does not authorise the clear:
`_on_safety_resume` still verifies the HMAC over `STRANDS_MESH_OVERRIDE_CODE`.

`tests/mesh/test_iot_policy_scope.py::TestSafetyCycleIsGrantedAsAPair` pins
Receive, Subscribe, Publish and the no-estop variant.
