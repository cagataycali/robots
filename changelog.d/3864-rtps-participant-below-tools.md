### Changed: the RTPS participant sits below both callers that publish through it

`RtpsRobot` opened a DDS writer by importing the `use_rtps` `@tool`, so a
library class reached up into the agent-tool layer for its transport. The
participant - one shared `DomainParticipant`, writers and readers cached per
`(topic, type)`, the IDL sample builder and the action dispatch - now lives in
`strands_robots.rtps.participant`, beside the mangling and IDL bundle it is
built on; `use_rtps` keeps what an agent envelope owns (the numeric-option
domains, the operator gate, the docstring a model reads). `rtps_action` takes
the operator gate as a required keyword argument, so no caller can publish to a
blocklisted surface by forgetting one, and both callers key it with the same
label, so a `cmd_vel` command reaching the same physical topic files one
interrupt id and one audit source whichever surface asked. No behaviour a
caller sees changes.
