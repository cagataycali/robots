### Fixed: `__repr__` no longer hides the constructor refusal that produced a half-built object

`repr` is what a traceback, a debugger and a failing assertion render. Ten
classes read instance attributes their `__init__` assigns only *after*
validating the caller's arguments, so a refusal left an instance whose
rendering reported `[AttributeError ... raised in repr()]` naming an attribute
unrelated to the refusal. `RosBridgedRobot(node_name="bad name!")` raised
`ValueError: invalid node_name: 'bad name!'`, and rendering the instance the
raising frame still held reported `'RosBridgedRobot' object has no attribute
'node_name'` - sending the reader after an attribute instead of the value they
had already passed.

`RosBridgedRobot`, `RosbridgeRobot`, `RtpsRobot`, `HardwareRtpsBridge`,
`InputPublisher`, `InputReceiver`, `Mesh`, `PeerInfo`, `DatasetRecorder` and
`ProcessorBridge` now report the lifecycle fact instead, through the new
`strands_robots.utils.partial_construction_repr`. It names no attribute, so
nobody is sent chasing one, and it owns the wording `IsaacSimulation` already
used - which now routes through it - so the phrase a reader recognises in a
traceback cannot diverge between layers. A fully constructed instance is
unaffected.
