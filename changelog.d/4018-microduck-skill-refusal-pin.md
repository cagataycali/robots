### Fixed: a microduck skill refusal is graded against the state the robot streams

The pin for `move` during a skill wrote `policy: "kick_left"` straight into the
driver's state cache, which the mock robotd overwrites with the literal's
`policy: "walk"` every 10 ms. It therefore had to win a 10 ms race, and on a
loaded runner it lost: the twist was allowed and the cell failed. `MockRobotd`
now carries the streamed `policy` as a mutable field, so a test puts the robot
into a skill and every refresh keeps saying so.
