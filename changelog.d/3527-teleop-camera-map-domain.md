### Fixed: a teleop camera map the lerobot argv cannot carry is refused

`lerobot_teleoperate`'s `robot_cameras` was rendered straight into the nested
`--robot.cameras` dict that lerobot parses, with nothing checked. `_build_camera_arg`
renders the default for every option an entry does not name, so `{"front": {"index": 4}}`
recorded camera **0** under the name `front`, `{"front": {"framerate": 60}}` recorded at
30 fps, and both reported `status="success"`. A geometry outside the domain
`lerobot_camera` reads the same pixels and frames with (`fps: 0`, `width: -640`) reached
the argv too, and a name or value carrying `,` `:` `{` `}` `=` or whitespace changed the
*shape* of that dict - one entry parsed as two, or a second camera the call never named.
A non-mapping entry answered `'str' object has no attribute 'get'`, which names nothing.

The whole map is now checked before a flag is rendered, at the single render site. Which
`type` values exist and which options each admits is read from lerobot's `CameraConfig`
registry through the one owner the `Robot` factory already uses
(`hardware_robot._camera_option_vocabulary`), so the `realsense` spelling the tool's own
schema suggested is refused naming `intelrealsense`, and a RealSense's
`serial_number_or_name` is admitted and rendered where the copied five-key list refused it.
A string value and the camera name are quoted in the rendered dict so draccus reads them
back verbatim: unquoted, `/dev/video[1]` failed to parse, a serial `0123` arrived as the
octal `"83"`, and a camera named `yes` or `null` was re-typed to the YAML 1.1 boolean or
null rather than kept as the dataset feature key the caller wrote. An integral
geometry or index is emitted as the whole number lerobot declares the field.
