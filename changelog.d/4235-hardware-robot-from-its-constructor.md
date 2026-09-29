### Tests: hardware `Robot` stand-ins start from the real constructor

Thirty-two test modules built a `strands_robots.hardware_robot.Robot` skeleton
through `__new__` and restated `__init__` field by field. They now start from
`tests._hardware_robot.hardware_robot_on(stand_in, ...)`, which runs the real
constructor with only its device-building step answered, and override only what
they model. A grader refuses a skeleton that restates a constructor default; a
bare `__new__` stays legal where partial construction is the subject. The
`cleanup()` error branch, until now reached only by skeletons failing at garbage
collection, has a pin of its own.
