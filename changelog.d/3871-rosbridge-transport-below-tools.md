### Changed: the rosbridge transport sits below both callers that dial through it

`RosbridgeRobot` opened a WebSocket by importing the `use_rosbridge` `@tool` -
and two of its private names with it - so a library class reached up into the
agent-tool layer for its transport. The transport itself (one long-lived
`roslibpy.Ros` per `(host, port)`, the `rosapi` graph introspection, the host /
port / name domains and the action dispatch) now lives in
`strands_robots.rosbridge`; `use_rosbridge` keeps what an agent envelope owns
(the numeric-option domains, the operator gate, the docstring a model reads).
`rosbridge_action` takes the operator gate as a required keyword argument, so no
caller can command a blocklisted surface by forgetting one, and both callers key
it with the same label, so a `cmd_vel` command reaching the same physical topic
files one interrupt id and one audit source whichever surface asked. No
behaviour a caller sees changes.
