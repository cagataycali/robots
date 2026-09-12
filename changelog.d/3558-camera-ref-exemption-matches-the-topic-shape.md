### Fixed: only a pointer topic gets the camera-frame drop exemption

The IoT transport drops `camera/` topics because a base64 JPEG frame does not
belong on MQTT, and exempts the small `camera/<cam>/ref` pointer that tells a
cloud subscriber the S3 key. The exemption was decided by sniffing substrings -
`topic.endswith("/ref") and "/camera/" in topic` - and a camera name is a free
string from the robot's own `config.cameras`, so a camera named `ref` published
its inline frame on `strands/<peer>/camera/ref`, which reads as a pointer: an
81 KB envelope reached the MQTT client past both the drop rule and MQTT's
128 KB cap, and the bridge republished it to the WAN. A `ref` tail under any
other family (`input/camera/ref`, `hand/camera/ref` - the 50 Hz LAN-only
topics) was exempted the same way. The exemption is now granted on the topic's
shape: the family segment must be `camera` and a camera name must sit between
it and the `ref` tail. A multi-segment camera name (`camera/front/left/ref`)
keeps its exemption, so no real pointer is refused. The bridge leg calls the
transport's own predicate instead of keeping a second copy of the rule, so the
two legs cannot drift on which topics are pointers.
