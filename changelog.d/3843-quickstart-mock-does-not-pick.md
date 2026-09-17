### Fixed: the quickstart promises what its own rollout does

The front matter read "Five minutes from install to a robot picking up a cube",
and the page's only rollout is `policy_provider="mock"`. `MockPolicy` declares
`reads_instruction = False` - it drives every joint through a test motion
whatever the task says - so the five minutes the page sold ended with the cube
where the reader put it: measured +1.5 mm in the page's own scene, all of it the
cube settling, 0.1 px of centroid motion over 300 rendered frames.

The library already refuses to let that read as a completed task: every task
envelope carries `instruction_not_read_notice`, added because "MockPolicy | wave
the arm ... completed" had been relayed as a wave that happened. The page was
making the same claim one level up, where no envelope reaches.

The description now says what the five minutes deliver, and the rollout section
says the cube has not moved and names the reference pick that lifts it on the
same `[sim-mujoco]` install (`examples/18_so101_pick_and_lift.py`, ~150 mm).
`tests/test_docs_quickstart_promise_matches_the_policy_it_runs.py` reads the
provider out of the page's own fence and requires the caveat only while that
provider's class declares it does not read the instruction, so a `mock` that
started acting on the words would retire the requirement rather than outlive it.
