### Fixed: the Go2's last sport-mode release is verified, not assumed to have failed

`Go2Driver.release_sport_mode` polls, because the release is asynchronous: the
robot keeps reporting `"ai"` until its onboard controller has actually let go of
the legs, so the driver calls `ReleaseMode()` and re-reads `CheckMode()` until the
reported mode name is empty. `attempts` is documented as how many
release-then-verify rounds to try.

The loop read once per round and released once per round, in that order, so the
last release had no read behind it. `attempts` rounds performed N releases but
only N-1 verified ones, and a robot that let go on its final attempt was reported
as `sport mode 'ai' still active after N release attempts` -- a statement about
the robot's present state that the driver had not asked it for since letting go.
Measured over a logging motion-switcher double, a mode clearing on the last
release was refused for every attempt count tried (1, 2 and 3). With `attempts=1`
the verb could never report success on a robot that was in a mode at all: it
released once, then gave up without looking.

The refusal is cached in `_sport_mode_released`, which is the write gate
`send_action`, `run_policy` and `start_task` consult, so the consequence is not
only a misleading message: the low-level write path stays shut on a released
robot until the caller happens to ask a second time, and the second call's first
read is the one that finally sees the release the first call performed.

N rounds now take N releases and N+1 reads -- one read to see what is holding the
robot, then one after every release, including the last. A refusal for a mode that
would not clear now names what the read taken *after* the final release reported.
The success path is unchanged: a robot already reporting no active mode still
releases nothing, and a mode that clears mid-loop is still confirmed by the read
that follows the release which cleared it.
