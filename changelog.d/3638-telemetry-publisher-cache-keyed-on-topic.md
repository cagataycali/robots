### Fixed: a telemetry publisher is now cached under its topic, not the caller's spelling

Both ROS 2 telemetry transports cached one publisher per robot (and per
robot/camera pair) so that a per-call `robot` argument could not send every
later robot's state out on the first one's topic - the failure a comment in
`HardwareRtpsBridge.__init__` already spelled out. But they keyed that cache on
the caller's *names* rather than on the topic those names select, and the
name-to-topic map is not one-to-one in either direction, so both halves of the
failure came back.

`RosTelemetryBase._safe` is documented as not injective: two names that differ
only in a run of separators render one token, because no valid ROS 2 name token
may carry the difference. Keyed on the name, each spelling advertised its own
publisher on the topic they share - one bridge appearing twice in
`ros2 topic info` for one camera. This is reachable with nothing exotic: a scene
holding cameras `arm0/wrist` and `arm0__wrist` (both spellings are legal, and
`camera_schema_key` names that very pair as the collision its dataset-side guard
exists to refuse) puts both keys in one observation, and `ros2_bridge=True`
published three image publishers for the two topics they name.

The image cache also joined `robot` and `camera` with `/`, a character either
name may contain, so one key could stand for two different topics. Measured on
both transports, `publish_image("arm", "wrist/rgb")` followed by
`publish_image("arm/wrist", "rgb")` created a single publisher: the second call
was handed the first's, and its frames went out on `/arm/wrist_rgb/image_raw` -
a topic it never named, while `/arm_wrist/rgb/image_raw` was never advertised at
all. Nothing reports that, because DDS matching is by topic name: the writer the
caller asked for simply never appears and no reader ever matches.

Both caches now key on the topic string, which is what identifies a publisher.
Two names selecting one topic share its single publisher, and two pairs naming
two topics get one publisher each. The change is the same in the rclpy and the
pure-RTPS bridge, so the two transports still put an identical graph up.
