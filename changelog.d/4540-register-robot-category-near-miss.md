### Fixed: `register_robot` names the group a near-miss category meant

`register_robot(category="arms")` stored the robot under a new one-robot
`"arms"` group beside the real `"arm"` one, where the docs catalog filter and
any caller looking up `"arm"` never found it, and said nothing. A category close
to a known group (`"arms"`, `"Arm"`, `"mobile-manip"`) is still registered as
given, and now logs a warning naming the group it probably meant. An exact
group, a deliberately new one (`"quadruped"`) and an empty category stay quiet.
